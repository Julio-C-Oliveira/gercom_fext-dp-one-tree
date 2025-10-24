from app.settings import aggregation_strategies, seeds, epsilons, number_of_rounds, results_folder
from app import utils

from app.server import Server

import time

def run_simulation():
    for strategy in aggregation_strategies:
        for seed in seeds:
            clients_dataset, inicialization_server_dataset, validation_server_dataset = utils.load_all_datasets(seed)

            results_file_path = f"{results_folder}{strategy}_{seed}.json"
            metrics = {}

            for epsilon in epsilons:
                metrics[epsilon] = {}

                for round in range(number_of_rounds):
                    metrics = [epsilon][round]

                    print(f"{strategy.upper()}\n | Random State: {seed:010d} | Epsilon: {epsilon:.2f} | Round: {round:02d} | Trees by Client: {trees_by_client:03d} |")

                    # Iniciar o servidor:
                    server = Server(inicialization_server_dataset, validation_server_dataset, epsilon)

                    # Iniciar e treinar os clientes:
                    clients_train_time = time.time()