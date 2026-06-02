import argparse
import time
import subprocess

from fedt.simulation.settings import simulation

def run_clients(number_of_clients_for_test, start_ID_for_clients):
    for setting in simulation.epsilon_settings:
        for strategy in simulation.aggregation_strategies:
            for i in range(simulation.number_of_simulations):
                print(f"Iniciando os clientes... Simulação: {i}")

                # Inicialização das métricas externas:
                # cpu_ram_proc = subprocess.Popen(
                #     [
                #         "fedt-cpu-ram", 
                #         "--strategy", f"{strategy}", 
                #         "--sim-number", f"{i}", "--user", "client"
                #     ]
                # )

                # net_proc = subprocess.Popen(
                #     [
                #         "fedt-network", 
                #         "--strategy", f"{strategy}", 
                #         "--sim-number", f"{i}", "--user", "client"
                #     ],
                #     stdout=subprocess.PIPE,
                #     text=True
                # )
                # tcpdump_output = net_proc.stdout.readline().strip()

                time.sleep(3)
                
                # Execução dos clientes:
                processes = []

                for i in range(number_of_clients_for_test):
                    cmd = [
                        "python", 
                        "-m", "fedt.app.client", 
                        "--client-id", str(i + int(start_ID_for_clients)),
                        "--number-of-rounds", str(1)
                    ]

                    p = subprocess.Popen(cmd)
                    processes.append(p)

                    time.sleep(5)

                for p in processes:
                    p.wait()

                # Encerrando as métricas externas:
                # tcpdump_processes = find_target_processes([tcpdump_output])
                # kill_processes(tcpdump_processes, "tcpdump")

                # cpu_ram_proc.wait()
                # net_proc.wait()
                print("Clientes finalizados, pausa de 30 segundos...")
                time.sleep(30)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Federated Learning for Decision Trees with Differential Privacy")
    parse.add_argument(
        "-n", "--number-of-clients-for-test",
        required=False,
        type=int,
        default=simulation.number_of_clients_for_test,
        help="Número de clients que devem ser executados nesse dispositivo."
    )
    parse.add_argument(
        "-i", "--start-id-for-clients",
        required=False,
        type=int,
        default=simulation.start_ID_for_clients,
        help="A base para o ID dos clientes que serão executados. Se for 0, os ids começarão em 0."
    )
    args = parse.parse_args()

    run_clients(
        number_of_clients_for_test = args.number_of_clients_for_test,
        start_ID_for_clients = args.start_id_for_clients
    )