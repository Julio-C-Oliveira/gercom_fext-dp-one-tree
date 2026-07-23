import argparse
import time
import subprocess

import numpy as np

from fedt.simulation.settings import simulation

def generate_random_numbers(number_of_seeds):
    rng = np.random.default_rng()
    return rng.integers(low=0, high=4294967296, size=number_of_seeds, dtype=np.uint64)

def run_server(seeds):
    for setting in simulation.epsilon_settings:
        for strategy in simulation.aggregation_strategies:
            for i, seed in enumerate(seeds):
                print(f"Iniciando o servidor... Simulação: {i}")
                print(f"\n[Epsilon: {setting.epsilon}] [Estratégia: {strategy}]")

                # Inicialização das métricas externas:
                # net_proc = subprocess.Popen(
                #     [
                #         "fedt-network", 
                #         "--strategy", f"{strategy}", 
                #         "--sim-number", f"{i}", 
                #         "--user", "server"
                #     ],
                #     stdout=subprocess.PIPE,
                #     text=True
                # )

                # tcpdump_output = net_proc.stdout.readline().strip()

                # time.sleep(0.5)

                # Execução do servidor:
                server_proc = subprocess.Popen(
                    [
                        "python", 
                        "-m", "fedt.app.server",
                        "--strategy", str(strategy),
                        "--epsilon", str(setting.epsilon),
                        "--number-of-clients", str(20), # Colocar pra CLI.
                        "--number-of-rounds", str(1), # Colocar pra CLI.
                        "--seed", str(seed),
                        "--threshold-type", str(setting.threshold_type),
                        "--threshold-value", str(setting.threshold_value),
                        "--threshold-multiplier", str(setting.threshold_multiplier)
                    ]
                )

                server_proc.wait()

                # Encerrando as métricas externas:
                # cpu_ram_proc = subprocess.Popen([
                #     "fedt-cpu-ram", 
                #     "--strategy", f"{strategy}", 
                #     "--sim-number", f"{i}", 
                #     "--user", "server", 
                #     "--pid", f"{server_proc.pid}"])

                # tcpdump_processes = find_target_processes([tcpdump_output])
                # kill_processes(tcpdump_processes, "tcpdump")

                # cpu_ram_proc.wait()
                # net_proc.wait()
                print("Server finalizado, pausa de 1 segundo...")
                time.sleep(1.5)

# TODO:
# - Ajustar os scripts
# - Ajustar a chamada do server

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Federated Learning for Decision Trees with Differential Privacy")
    parse.add_argument(
        "-s", "--run-with-seeds",
        required=False,
        action="store_true",
        help="Utilize essa flag para definir que as seeds devem ser utilizadas."
    )
    args = parse.parse_args()

    if not args.run_with_seeds:
        seeds = generate_random_numbers(simulation.number_of_simulations)
    else:
        seeds = simulation.seeds

    run_server(seeds)