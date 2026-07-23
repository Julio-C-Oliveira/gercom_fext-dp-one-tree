import asyncio
import time
import os
import json
import gc 

import grpc
import grpc.aio as grpc_aio

from fedt.app.settings import settings
from fedt.app.settings import paths

from fedt.app import utils
from fedt.app.utils import create_specific_result_folder
from fedt.app.utils import format_time
from fedt.service import fedT_pb2
from fedt.service import fedT_pb2_grpc

from sklearn.ensemble import RandomForestRegressor
from fedt.app.client_utils import Client
from fedt.app.utils import get_final_seed

import argparse
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(max_workers=None)

def send_stream_trees(serialise_trees:bytes, client_ID:int):
    async def _gen():
        for tree in serialise_trees:
            msg = fedT_pb2.Forest_CLient()
            msg.client_ID = client_ID
            msg.serialised_tree = tree
            yield msg
            await asyncio.sleep(0)
    return _gen()

async def run():
    async with grpc_aio.insecure_channel(f"{settings.server.IP}:{settings.server.port}") as channel:
        stub = fedT_pb2_grpc.FedTStub(channel)

        dataset = None

        for round_idx in range(number_of_rounds): # Gambiarra momentanea, irei adapatar para usar o cli.
            round_start_time = time.time()
            logger.warning(f"Round: {round_idx}")

            request_settings = fedT_pb2.Request_Server(client_ID=ID)
            server_reply_settings = await stub.get_server_settings(request_settings)
            
            current_round = server_reply_settings.current_round
            if server_reply_settings.HasField('seed'):
                seed = server_reply_settings.seed.value
            else:
                seed = None
            epsilon = server_reply_settings.epsilon

            server_round = getattr(server_reply_settings, "current_round", None)

            seed = get_final_seed(ID, seed)

            logger.debug(f"Round Atual: {current_round}.")
            logger.debug(f"Seed: {seed}.")
            logger.debug(f"Epsilon: {epsilon}.")

            wait_start = time.time()
            while server_round is not None and server_round < round_idx:
                logger.info(f"Servidor no round {server_round}, esperando atingir round {round_idx}...")
                await asyncio.sleep(5)
                server_reply_settings = await stub.get_server_settings(request_settings)

                current_round = server_reply_settings.current_round
                if server_reply_settings.HasField('seed'):
                    seed = server_reply_settings.seed.value
                else:
                    seed = None
                epsilon = server_reply_settings.epsilon

                seed = get_final_seed(ID, seed)

                if time.time() - wait_start > settings.client.timeout:
                    raise RuntimeError(f"[Client {ID}] Timeout esperando servidor avançar do round {server_round} para {round_idx}")

            if dataset is None:
                dataset = utils.load_house_client(
                    seed=seed,
                    alpha=dirichlet_alpha,
                    bins=number_of_bins_for_dirichlet
                )

            request_model = fedT_pb2.Request_Server(client_ID=ID)
            server_trees_serialised = []
            async for server_reply in stub.get_server_model(request_model):
                server_trees_serialised.append(server_reply.serialised_tree)

            first_server_serialise_trees_size = utils.get_size_of_many_serialised_models(server_trees_serialised)
            logger.debug(f"Early Server Model in MB: {first_server_serialise_trees_size/(1024**2)}")

            loop = asyncio.get_running_loop()
            server_trees_deserialise = await loop.run_in_executor(
                executor,
                utils.deserialise_several_trees,
                server_trees_serialised
            )
            del server_trees_serialised
            gc.collect()

            server_model = RandomForestRegressor(
                n_estimators=settings.number_of_clients,
                max_depth=3,
                warm_start=True,
                random_state=seed
            )
            server_model.fit(dataset[0], dataset[1])
            server_model.estimators_ = server_trees_deserialise

            fit_start_time = time.time()
            client = Client(
                dataset, 
                ID, 
                seed, 
                epsilon
            )
            fit_time = time.time() - fit_start_time
            
            initial_evaluate_metrics = client.choose_model(
                global_model=server_model,
                update_local_model=False
            )
            logger.info(f"\nModelo Inicial:\nMean Squared Error: {initial_evaluate_metrics["mse"]:.3f}\nPearson: {initial_evaluate_metrics["pearson"]:.3f}")

            serialise_tree = await loop.run_in_executor(
                executor,
                utils.serialise_tree,
                client.local_model
            )
            client_serialise_tree_size = utils.get_serialised_size_bytes(serialise_tree)
            logger.debug(f"Local Model in MB: {client_serialise_tree_size/(1024**2)}")

            aggregate_trees_request = fedT_pb2.Client_Tree()
            aggregate_trees_request.client_ID = ID
            aggregate_trees_request.serialised_tree = serialise_tree

            server_trees_serialised = []
            async for reply in stub.aggregate_trees(aggregate_trees_request):
                server_trees_serialised.append(reply.serialised_tree)

            del serialise_tree
            gc.collect()

            logger.info("Modelo global recebido")

            server_trees_deserialised = await loop.run_in_executor(
                executor,
                utils.deserialise_several_trees,
                server_trees_serialised
            )
            server_model.estimators_ = server_trees_deserialised

            final_server_serialise_trees_size = utils.get_size_of_many_serialised_models(server_trees_serialised)
            logger.debug(f"Final Server Model in MB: {final_server_serialise_trees_size/(1024**2)}")

            evaluate_start_time = time.time()
            final_evaluate_metrics = await loop.run_in_executor(
                executor,
                client.choose_model,
                server_model
            )
            evaluate_time = time.time() - evaluate_start_time
            logger.info(f"\nModelo Final:\nMean Squared Error: {final_evaluate_metrics["mse"]:.3f}\nPearson: {final_evaluate_metrics["pearson"]:.3f}")

            round_end_time = time.time()
            round_time = round_end_time - round_start_time

            start_inference_time = time.time()
            await loop.run_in_executor(
                executor,
                client.evaluate_inference_time,
                100
            )
            inference_time = time.time() - start_inference_time
            logger.debug(f"\nDuração do Round: {format_time(round_time)}\nTempo de treinamento: {format_time(fit_time)}\nTempo de avaliação: {format_time(evaluate_time)}\nTempo de inferência: {format_time(inference_time)}")

            request_end = fedT_pb2.Request_End(
                client_ID = ID,
                client_tree_size = client_serialise_tree_size,
                server_tree_size = final_server_serialise_trees_size,
                fit_time = fit_time,
                initial_mse = initial_evaluate_metrics["mse"],
                initial_pearson = initial_evaluate_metrics["pearson"],
                final_mse = final_evaluate_metrics["mse"],
                final_pearson = final_evaluate_metrics["pearson"],
                round_time = round_time,
                round_start_time = round_start_time, 
                round_end_time = round_end_time,
                evaluate_time = evaluate_time,
                inference_time = inference_time
            )
            await stub.end_of_transmission(request_end)

            del server_model, client, server_trees_serialised, server_trees_deserialised

            gc.collect()
            await asyncio.sleep(0.1)


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
        "-a", "--global-max-target",
        required=False,
        type=int,
        default=settings.differential_privacy.global_max_target,
        help="O Maior valor que será aceito pelo modelo sem ser truncado."
    )
    parse.add_argument(
        "-i", "--global-min-target",
        required=False,
        type=int,
        default=settings.differential_privacy.global_min_target,
        help="O Menor valor que será aceito pelo modelo sem ser truncado."
    )
    parse.add_argument(
        "-d", "--max-tree-depth",
        required=False,
        type=int,
        default=settings.differential_privacy.tree_max_depth,
        help="A profundidade máxima das árvores."
    )
    parse.add_argument(
        "-t", "--timeout",
        required=False,
        type=int,
        default=settings.client.timeout,
        help="O tempo que o cliente vai esperar por respostas do servidor."
    )
    parse.add_argument(
        "-g", "--debug",
        required=False,
        action="store_true",
        help="Essa flag serve para habilitar os logs de debug."
    )
    parse.add_argument(
        "-e", "--evaluate-type",
        required=False,
        type=str,
        default=settings.client.evaluate_type,
        choices=["mse", "pearson"],
        help="A métrica que o cliente irá utilizar para avaliar o modelo."
    )
    parse.add_argument(
        "-r", "--number-of-rounds",
        required=False,
        type=int,
        default=settings.number_of_rounds,
        help="Números de rounds que serão executados."
    )
    parse.add_argument(
        "-l", "--dirichlet-alpha",
        required=False,
        type=float,
        default=settings.client.dirichlet_alpha,
        help="O alpha que será utilizado para gerar a distribuição Non-IID."
    )
    parse.add_argument(
        "-s", "--number-of-bins-for-dirichlet",
        required=False,
        type=int,
        default=settings.client.number_of_bins_for_dirichlet,
        help="O número de bins em que a distribuição do dataset será dividida."
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
    parse.add_argument(
        "-c", "--client-id",
        required=False,
        type=int,
        default=0,
        help="Client ID, os clientes devem ter IDs distintos."
    )
    args = parse.parse_args()

    ID = args.client_id
    number_of_rounds = args.number_of_rounds
    dirichlet_alpha = args.dirichlet_alpha
    number_of_bins_for_dirichlet = args.number_of_bins_for_dirichlet

    log_level = logging.DEBUG if True else logging.INFO
    logger = utils.setup_logger(
        name=f"Client {ID}",
        log_file=f"fedt_client_{ID}.log",
        level=log_level
    )

    asyncio.run(run())
    executor.shutdown(wait=True)