import argparse
import time
import subprocess

from fedt.simulation.settings import simulation

def run_server():
    for epsilon in simulation.epsilons:
        for strategy in simulation.aggregation_strategies:
            for i in range(simulation.number_of_simulations):
                print(f"Iniciando o servidor... Simulação: {i}")

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

                time.sleep(3)

                # Execução do servidor:
                server_proc = subprocess.Popen(
                    [
                        "python", 
                        "-m", "fedt.app.server",
                        "--strategy", str(strategy),
                        "--epsilon", str(epsilon)
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
                print("Server finalizado, pausa de 10 segundos...")
                time.sleep(10)

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

    run_server()