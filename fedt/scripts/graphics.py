import os
import json
import numpy as np
from collections import defaultdict

def carregar_dados_simulacao(base_path, target_metric="final_rmse", user_type="clients"):
    """
    Percorre a estrutura de diretórios, carrega os arquivos JSON e extrai as métricas.
    Retorna um dicionário aninhado: dados[estrategia][epsilon] = [lista_de_valores]
    """
    # Dicionário para armazenar os dados separados por estratégia e epsilon
    dados_agregados = defaultdict(lambda: defaultdict(list))
    
    # 1. Identifica as pastas das estratégias dentro de results/resultado_01-06/
    estrategias = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for estrategia in estrategias:
        server_path = os.path.join(base_path, estrategia, "server")
        
        if not os.path.exists(server_path):
            continue
            
        # Lista todos os arquivos JSON daquela estratégia
        arquivos_json = [f for f in os.listdir(server_path) if f.endswith(".json")]
        
        for arquivo in arquivos_json:
            # Ex: all_trees_0.5_10.json -> remove o ".json"
            nome_sem_extensao = arquivo.replace(".json", "")
            
            # Remove o prefixo da estratégia para isolar o epsilon e a execução
            # Ex: "all_trees_0.5_10" -> "0.5_10"
            prefixo_len = len(estrategia) + 1 # +1 por causa do "_"
            resto_do_nome = nome_sem_extensao[prefixo_len:]
            
            # Divide sempre pelo último "_" para separar o epsilon do número da execução
            # Ex: "no-diff-privacy_10" -> ["no-diff-privacy", "10"]
            partes = resto_do_nome.rsplit("_", 1)
            if len(partes) != 2:
                continue
                
            epsilon = partes[0]
            
            # 2. CARREGAMENTO DOS DADOS (Leitura explícita do JSON)
            caminho_arquivo = os.path.join(server_path, arquivo)
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                try:
                    dados_json = json.load(f)
                except json.JSONDecodeError:
                    print(f"Erro ao ler o arquivo {arquivo}. Pulando...")
                    continue
            
            # 3. Lógica para pegar o último round
            rounds = [k for k in dados_json.keys() if k.startswith("round_")]
            if not rounds:
                continue
            
            # Ordena numericamente ("round_1", "round_2", ..., "round_10")
            rounds.sort(key=lambda x: int(x.split("_")[1]))
            ultimo_round = rounds[-1]
            
            # 4. Extração da métrica desejada e armazenamento
            if user_type == "clients":
                clients_data = dados_json[ultimo_round].get("clients", {})
                valores = [c_data[target_metric] for c_data in clients_data.values() if target_metric in c_data]
                if valores:
                    dados_agregados[estrategia][epsilon].append(np.mean(valores))
            
            elif user_type == "server":
                server_data = dados_json[ultimo_round].get("server", {})
                if target_metric in server_data:
                    dados_agregados[estrategia][epsilon].append(server_data[target_metric])
                    
    return dados_agregados


def exibir_resultados_separados(dados_agregados, metric_name="final_rmse"):
    """
    Exibe os dados no terminal separando primeiro por Estratégia e depois por Privacidade (Epsilon).
    """
    print("\n" + "="*70)
    print(f" RESULTADOS AGREGADOS - Métrica: {metric_name}")
    print("="*70)
    
    # Função auxiliar para ordenar os Epsilons numericamente e deixar o "no-diff-privacy" no final
    def order_epsilon(e):
        if e == "no-diff-privacy":
            return float('inf')
        return float(e)

    # Itera separadamente por cada estratégia carregada
    for estrategia, epsilons_dict in dados_agregados.items():
        print(f"\n{'='*70}")
        print(f" >>> ESTRATÉGIA: {estrategia.upper()}")
        print(f"{'='*70}")
        print(f"{'Nível de Privacidade (Epsilon)':<35} | {'Média':<12} | {'Desvio Padrão'}")
        print("-" * 70)
        
        # Itera por cada nível de privacidade (epsilon)
        for epsilon in sorted(epsilons_dict.keys(), key=order_epsilon):
            valores_execucoes = epsilons_dict[epsilon]
            
            media = np.mean(valores_execucoes)
            desvio = np.std(valores_execucoes)
            
            # Exibe os dados agrupados por epsilon
            print(f"{epsilon:<35} | {media:<12.4f} | {desvio:<12.4f}")
            
    print("\n")


if __name__ == "__main__":
    # Caminho que você mostrou no tree
    caminho_base = "results/resultado_01-06"
    
    # 1. Carrega os dados na memória passando a métrica desejada
    dados_processados = carregar_dados_simulacao(
        base_path=caminho_base, 
        target_metric="initial_mse", # Altere aqui para a métrica real que quer analisar
        user_type="clients"
    )
    
    # 2. Exibe os resultados separados conforme solicitado
    if dados_processados:
        exibir_resultados_separados(dados_processados, metric_name="final_rmse")
    else:
        print("Nenhum dado foi encontrado ou a métrica especificada não existe nos JSONs.")