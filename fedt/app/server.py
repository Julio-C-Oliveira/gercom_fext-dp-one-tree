import asyncio
import logging
import time
import json
import os
import gc

import grpc
import grpc.aio as grpc_aio

from sklearn.ensemble import RandomForestRegressor
import numpy as np

from fedt.app.settings import settings, paths

from fedt.app.server_strategy import Strategy
from fedt.app import utils
from fedt.app.utils import create_specific_result_folder
from fedt.service import fedT_pb2
from fedt.service import fedT_pb2_grpc
from google.protobuf import wrappers_pb2

import warnings
from scipy.stats import ConstantInputWarning

from concurrent.futures import ThreadPoolExecutor

import argparse

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# Configuração do log:
log_level = logging.DEBUG if True else logging.INFO
logger = utils.setup_logger(
    name="SERVER",
    log_file="fedt_server.log",
    level=log_level
)

def add_end_time(runtime_clients, ID, end_time):
    for i, (client_id, start_time) in enumerate(runtime_clients):
        if client_id == ID:
            runtime_clients[i] = (client_id, (start_time, end_time))
            break
    return runtime_clients

def average_runtime(runtime_clients):
    """Calcula o tempo médio de execução."""
    runtime_list = [(end - start) for (_, (start, end)) in runtime_clients]
    runtime_sum = sum(runtime_list)
    runtime_average = runtime_sum / settings.number_of_clients
    return runtime_average

# Falta as funções externas se adaptarem ao number of clients interno
class FedT(fedT_pb2_grpc.FedTServicer):
    def __init__(
        self, 
        number_of_jobs,
        number_of_clients,
        number_of_rounds,
        seed,
        strategy,
        epsilon, 
        beta,
        threshold_type,
        threshold_value,
        threshold_multiplier
        ) -> None:

        super().__init__()

        self.number_of_rounds = number_of_rounds

        self.aggregation_strategy = strategy
        self.seed = seed
        self.epsilon = epsilon

        self.lock = asyncio.Lock()
        self.aggregation_done = asyncio.Event()

        self.round = 0
        self.aggregation_realised = 0 # 0 waiting, 1 aggregating, 2 done.

        self.clientes_conectados = []
        self.clientes_esperados = number_of_clients
        self.clientes_respondidos = 0
        self.trees_warehouse = []
        self.runtime_clients = []
        self.aggregation_time = 0.0

        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.threshold_multiplier = threshold_multiplier

        self._supervisor_started = False
        self.shutdown_event = None

        self.executor = ThreadPoolExecutor(max_workers=number_of_jobs)

        self.global_model = RandomForestRegressor( # Dar um jeito de nem precisar treinar um modelo.
            n_estimators=self.clientes_esperados,
            max_depth=3,
            warm_start=True,
            random_state=self.seed
        )
        data_train, label_train = utils.load_dataset_for_server(self.seed)
        self.global_model.fit( # Colocar na chamada da função.
            data_train, label_train
        )

        self.validation_dataset = utils.load_server_side_validation_data(
            seed=utils.get_final_seed(self.clientes_esperados, self.seed)
        )

        self.global_trees = self.global_model.estimators_

        self.current_round_clients_data = {}
        self.all_execution_data = {}

        if epsilon >= 0:
            base_file_name = f"{self.aggregation_strategy}_{epsilon}"
        else:
            base_file_name = f"{self.aggregation_strategy}_no-diff-privacy"
        new_results_folder = create_specific_result_folder(
            results_folder=paths.results_folder,
            strategy=self.aggregation_strategy,
            base_name="server"
        )
        existing_files = [
            file for file in os.listdir(new_results_folder)
            if file.startswith(base_file_name) and file.endswith(".json")
        ]

        next_file_index = len(existing_files) + 1
        result_file_name = f"{base_file_name}_{next_file_index}.json"
        self.server_result_path = (new_results_folder / result_file_name).resolve()

        logger.warning(f"Result path: {self.server_result_path}")

    def attach_shutdown_event(self, event):
        self.shutdown_event = event

    def aggregate_strategy(self, received_trees): # Tenho que adaptar isso à troca de pearson.
        match self.aggregation_strategy:
            case "all_trees":
                self.global_model.estimators_ = Strategy.all_trees(received_trees)
            case "threshold_trees": 
                self.global_model.estimators_ = Strategy.threshold_trees(
                    self.validation_dataset, 
                    received_trees, 
                    self.threshold_type, self.threshold_value, self.threshold_multiplier
                )
            case "merge_trees":
                merged = Strategy.merge_trees(
                    received_trees=received_trees,
                    max_depth_global=settings.differential_privacy.tree_max_depth,
                    seed=self.seed,
                )
                # Armazena a árvore fundida como único estimador do ensemble
                self.global_model.estimators_ = [merged]
            case _:
                self.global_model.estimators_ = Strategy.all_trees(received_trees)

    async def _supervisor_task(self):
        while True:
            await asyncio.sleep(0.2)

            async with self.lock:
                enough = ( len(self.trees_warehouse) >= self.clientes_esperados )
                should_start = ( self.aggregation_realised == 0 and enough )

                if should_start:
                    self.aggregation_realised = 1
                    break

        logger.info(f"Supervisor iniciando agregação, round {self.round}")

        forests = [trees for (_, trees) in self.trees_warehouse]
        start_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self.aggregate_strategy, forests)

            self.aggregation_time = time.time() - start_time
            logger.info(f"Agregação finalizada para o round {self.round}")
        except Exception as error:
            logger.critical(f"Erro na agregação: {error}")

        async with self.lock:
            self.aggregation_realised = 2
            self.aggregation_done.set()


    async def aggregate_trees(self, request, context): # Retirar o request iterator.
        client_ID = None

        logger.info(f"Recebendo as árvores dos clientes, Round: {self.round}")

        client_ID = request.client_ID

        loop = asyncio.get_running_loop()
        client_tree = await loop.run_in_executor(
            self.executor,
            utils.deserialise_tree,
            request.serialised_tree
        )
        
        async with self.lock:
            if client_ID not in self.clientes_conectados:
                self.clientes_conectados.append(client_ID)
            self.trees_warehouse.append((client_ID, client_tree))

            logger.debug(f"O cliente {client_ID} enviou sua árvore.")
            logger.info(f"Clientes conectados {len(self.clientes_conectados)}/{self.clientes_esperados}")

            if not self._supervisor_started:
                self._supervisor_started = True
                asyncio.create_task(self._supervisor_task())

        await self.aggregation_done.wait()

        serialised_global_trees = await loop.run_in_executor(
            self.executor, 
            utils.serialise_several_trees, 
            self.global_model.estimators_
        )
        number_of_trees = len(serialised_global_trees)
        number_of_sended_trees = 0

        server_reply = fedT_pb2.Forest_Server()
        for tree in serialised_global_trees:
            number_of_sended_trees += 1
            if number_of_sended_trees % settings.server.print_every_trees_sent == 0:
                logger.info(f"Client ID: {client_ID}. Àrvore {number_of_sended_trees} de {number_of_trees} enviada.")
            server_reply.serialised_tree = tree
            yield server_reply

    async def get_server_model(self, request, context):
        start_time = time.time()

        self.runtime_clients.append([request.client_ID, start_time])
        logger.info(f"Client ID: {request.client_ID}, requisitando o modelo do servidor.")
        
        loop = asyncio.get_running_loop()
        trees = utils.get_model_parameters(self.global_model)
        serialised_trees = await loop.run_in_executor(
            self.executor, 
            utils.serialise_several_trees, 
            trees
        )
        
        server_message = fedT_pb2.Forest_Server()
        for serialise_tree in serialised_trees:
            server_message.serialised_tree = serialise_tree
            yield server_message

    async def get_server_settings(self, request, context):
        logger.debug(f"Client ID: {request.client_ID}, solicitando as configurações.")
        return fedT_pb2.Server_Settings(
            current_round=self.round,
            seed=wrappers_pb2.UInt64Value(value=self.seed) if self.seed is not None else None,
            epsilon=self.epsilon
        )

    def _calculate_aggregated_metrics(self):
        clients_data = self.current_round_clients_data.values()
        
        if not clients_data:
            return {}

        metrics_to_aggregate = [
            "client_tree_size", "server_tree_size", "fit_time",
            "initial_rmse", "initial_mse", "final_rmse", "final_mse",
            "round_time", "evaluate_time", "inference_time"
        ]

        aggregated_metrics = {}

        # Calcula média e desvio padrão
        for metric in metrics_to_aggregate:
            values = [client[metric] for client in clients_data if metric in client]
            if values:
                aggregated_metrics[f"{metric}_mean"] = float(np.mean(values))
                aggregated_metrics[f"{metric}_std"] = float(np.std(values))
            else:
                aggregated_metrics[f"{metric}_mean"] = None
                aggregated_metrics[f"{metric}_std"] = None

        # Calcula o menor start_time e o maior end_time
        start_times = [client["round_start_time"] for client in clients_data if "round_start_time" in client]
        end_times = [client["round_end_time"] for client in clients_data if "round_end_time" in client]

        aggregated_metrics["round_start_time_min"] = float(min(start_times)) if start_times else None
        aggregated_metrics["round_end_time_max"] = float(max(end_times)) if end_times else None

        return aggregated_metrics

    def save_round_results(self):
        # Calcular as métricas agregadas dos clientes:
        aggregated_client_metrics = self._calculate_aggregated_metrics()

        # Métricas do servidor:
        server_metrics = {
            "average_client_runtime": average_runtime(self.runtime_clients),
            "aggregation_time": self.aggregation_time,
            **aggregated_client_metrics
        }

        # Métricas do cliente:
        self.all_execution_data[f"round_{self.round}"] = {
            "server": server_metrics,
            "clients": self.current_round_clients_data.copy()
        }

        # Escrita:
        try:
            with open(self.server_result_path, "w", encoding="utf-8") as f:
                json.dump(self.all_execution_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Resultados do Round {self.round} salvos com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo JSON: {e}")

        # Limpa o buffer
        self.current_round_clients_data.clear()


    async def end_of_transmission(self, request, context):
        end_time = time.time()
        async with self.lock:
            self.runtime_clients = add_end_time(
                self.runtime_clients, 
                request.client_ID, 
                end_time
            )
            self.clientes_respondidos += 1
            logger.info(f"O cliente {request.client_ID} finalizou round. Clientes respondidos: {self.clientes_respondidos}/{self.clientes_esperados}")

            logger.info(f"Registrando as métricas do client: {request.client_ID}")
            self.current_round_clients_data[f"client_{request.client_ID}"] = {
                "client_tree_size": request.client_tree_size,
                "server_tree_size": request.server_tree_size,
                "fit_time": request.fit_time,
                "initial_rmse": np.sqrt(request.initial_mse),
                "initial_mse": request.initial_mse,
                "initial_pearson": request.initial_pearson,
                "final_rmse": np.sqrt(request.final_mse),
                "final_mse": request.final_mse,
                "final_pearson": request.final_pearson,
                "round_time": request.round_time,
                "round_start_time": request.round_start_time,
                "round_end_time": request.round_end_time,
                "evaluate_time": request.evaluate_time,
                "inference_time": request.inference_time
            }

            if self.clientes_respondidos == self.clientes_esperados:
                logger.info("Todos os clientes finalizaram.")

                self.save_round_results()

                for i in self.runtime_clients:
                    logger.debug(f"Client ID: {i[0]} → tempo de execução: {utils.format_time(i[1][1] - i[1][0])}")

                logger.info(f"Tempo de Execução Médio: {utils.format_time(average_runtime(self.runtime_clients))}")

                await self._reset_server_async()

                logger.warning(f"Round {self.round} finalizado")
                self.round += 1

                if self.round >= self.number_of_rounds:
                    logger.warning(f"Encerrando treinamento em 5 segundos...")
                    self.shutdown_event.set()
                    return fedT_pb2.OK(ok=1)
                else: 
                    self.aggregation_realised = 0
                    self.aggregation_done = asyncio.Event()
                    self._supervisor_started = False

                    logger.warning(f"Round {self.round} iniciado")

        return fedT_pb2.OK(ok=1)

    async def _reset_server_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._reset_server_sync)

    def _reset_server_sync(self):
        logger.warning("Resetando estado do servidor...")
        
        del self.global_model, self.global_trees
        gc.collect()

        self.global_model = RandomForestRegressor(
            n_estimators=self.clientes_esperados,
            max_depth=3,
            warm_start=True
        )
        data_train, label_train = utils.load_dataset_for_server(self.seed)
        self.global_model.fit(data_train, label_train)

        self.global_trees = self.global_model.estimators_

        self.clientes_conectados = []
        self.clientes_respondidos = 0
        self.trees_warehouse = []
        self.aggregation_realised = 0
        self.runtime_clients = []
        self.aggregation_time = 0.0

    
async def run_server(parse_args):
    logger.info("Servidor inicializando...")

    server = grpc_aio.server()
    servicer = FedT(
        number_of_jobs=parse_args.number_of_jobs,
        number_of_clients=parse_args.number_of_clients,
        number_of_rounds=parse_args.number_of_rounds,
        seed=parse_args.seed,
        strategy=parse_args.strategy,
        epsilon=parse_args.epsilon,
        beta=parse_args.beta,
        threshold_type=parse_args.threshold_type,
        threshold_value=parse_args.threshold_value,
        threshold_multiplier=parse_args.threshold_multiplier
    )

    shutdown_event = asyncio.Event()
    servicer.attach_shutdown_event(shutdown_event)

    fedT_pb2_grpc.add_FedTServicer_to_server(servicer, server)

    server.add_insecure_port(f"{parse_args.ip}:{parse_args.port}")
    
    await server.start()
    logger.info(f"Servidor ativo - {parse_args.ip}:{parse_args.port}")

    await shutdown_event.wait()
    logger.warning("Shutdown event recebido, desligando o servidor...")

    await server.stop(grace=10)
    await server.wait_for_termination()

    servicer.executor.shutdown(wait=True)
    logger.warning("Servidor encerrado.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Federated Learning for Decision Trees with Differential Privacy")
    parse.add_argument(
        "-j", "--number-of-jobs",
        required=False,
        type=int,
        default=settings.number_of_jobs,
        help="Números de núcleos que podem ser utilizados."
    )
    parse.add_argument(
        "-c", "--number-of-clients",
        required=False,
        type=int,
        default=settings.number_of_clients,
        help="Números de clientes esperados."
    )
    parse.add_argument(
        "-r", "--number-of-rounds",
        required=False,
        type=int,
        default=settings.number_of_rounds,
        help="Números de rounds que serão executados."
    )
    parse.add_argument(
        "-s", "--seed",
        required=False,
        type=int,
        default=settings.seed,
        help="Random State, para produzir reprodutibilidade."
    )
    parse.add_argument(
        "-t", "--strategy",
        required=False,
        type=str,
        default=settings.aggregation_strategy,
        choices=["all_trees", "threshold_trees", "merge_trees"],
        help="A estrátegia à ser utilizada."
    )
    parse.add_argument(
        "-e", "--epsilon",
        required=False,
        type=float,
        default=settings.differential_privacy.epsilon,
        help="Nível de privacidade, quanto menor o epsilon, maior o nível de privacidade aplicado."
    )
    parse.add_argument(
        "-b", "--beta",
        required=False,
        type=float,
        default=settings.differential_privacy.balancing_coefficient,
        help="Coeficiente de balanceamento, serve para controlar a distribuição de orçamento entre camadas internas e nós folha. Definido entre 0 e 1."
    )
    parse.add_argument(
        "-y", "--threshold-type",
        required=False,
        type=str,
        default=settings.server.threshold_type,
        help="O tipo de limiar que será utilizado na estrátegia threshold trees."
    )
    parse.add_argument(
        "-v", "--threshold-value",
        required=False,
        type=float,
        default=settings.server.threshold_value,
        help="O valor de limiar que será utilizado na estrátegia threshold trees."
    )
    parse.add_argument(
        "-m", "--threshold-multiplier",
        required=False,
        type=float,
        default=settings.server.threshold_multiplier,
        help="O valor que irá ser multiplicado pelo limiar, caso nenhuma árvore seja selecionada, serve para ajustar o limiar."
    )
    parse.add_argument(
        "-o", "--timeout",
        required=False,
        type=int,
        default=settings.server.timeout,
        help="O tempo que o server vai esperar por respostas dos clientes."
    )
    parse.add_argument(
        "-g", "--debug",
        required=False,
        action="store_true",
        help="Essa flag serve para habilitar os logs de debug."
    )
    parse.add_argument(
        "--ip",
        required=False,
        type=str,
        default=settings.server.IP,
        help="O IP do servidor."
    )
    parse.add_argument(
        "--port",
        required=False,
        type=int,
        default=settings.server.port,
        help="A porta do servidor."
    )

    asyncio.run(
        run_server(
            parse_args=parse.parse_args()
        )
    )