import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def carregar_dados_simulacao(base_path, target_metric="final_rmse", user_type="clients"):
    """
    Carrega os dados considerando apenas o último (ou único) round disponível.
    """
    dados_agregados = defaultdict(lambda: defaultdict(list))
    estrategias = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for estrategia in estrategias:
        server_path = os.path.join(base_path, estrategia, "server")
        if not os.path.exists(server_path):
            continue
            
        arquivos_json = [f for f in os.listdir(server_path) if f.endswith(".json")]
        
        for arquivo in arquivos_json:
            nome_sem_extensao = arquivo.replace(".json", "")
            prefixo_len = len(estrategia) + 1 
            resto_do_nome = nome_sem_extensao[prefixo_len:]
            
            partes = resto_do_nome.rsplit("_", 1)
            if len(partes) != 2:
                continue
                
            epsilon = partes[0]
            caminho_arquivo = os.path.join(server_path, arquivo)
            
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                try:
                    dados_json = json.load(f)
                except json.JSONDecodeError:
                    continue
            
            rounds = [k for k in dados_json.keys() if k.startswith("round_")]
            if not rounds:
                continue
            
            rounds.sort(key=lambda x: int(x.split("_")[1]))
            ultimo_round = rounds[-1] # Pega o round_0
            
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

def plotar_impacto_epsilon(dados_agregados, estrategia_alvo, metric_name):
    """
    Plota um gráfico de linha onde o Eixo X é o Epsilon e o Eixo Y é a Métrica.
    Ordenação invertida: Do menos privado (Sem Priv.) ao mais privado (Epsilon menor).
    """
    if estrategia_alvo not in dados_agregados:
        print(f"Estratégia '{estrategia_alvo}' não encontrada para gráfico de linha.")
        return

    dados_estrategia = dados_agregados[estrategia_alvo]
    
    def order_epsilon(e):
        if e == "no-diff-privacy":
            return float('inf')
        return float(e)

    # reverse=True inverte a ordem: Infinito (Sem Priv.) vem primeiro, seguido dos maiores epsilons até os menores
    epsilons_ordenados = sorted(dados_estrategia.keys(), key=order_epsilon, reverse=True)
    
    medias = []
    desvios = []
    labels_x = []

    for eps in epsilons_ordenados:
        valores = dados_estrategia[eps]
        medias.append(np.mean(valores))
        desvios.append(np.std(valores))
        labels_x.append("Sem Priv." if eps == "no-diff-privacy" else str(eps))

    plt.figure(figsize=(9, 6))
    
    plt.errorbar(labels_x, medias, yerr=desvios, fmt='-o', color='b', 
                 capsize=5, capthick=2, elinewidth=2, markersize=8, 
                 label=f'Estratégia: {estrategia_alvo}')

    plt.title(f"Impacto da Privacidade na Métrica {metric_name.upper()}", fontsize=14)
    plt.xlabel("Nível de Privacidade (Epsilon)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{metric_name}_linha.pdf")
    plt.close()


def plotar_impacto_epsilon_boxplot(dados_agregados, estrategia_alvo, metric_name):
    """
    Gera um gráfico Boxplot com ordenação invertida (Sem Priv. até Epsilon menor).
    """
    if estrategia_alvo not in dados_agregados:
        print(f"Estratégia '{estrategia_alvo}' não encontrada para Boxplot.")
        return

    dados_estrategia = dados_agregados[estrategia_alvo]
    
    def order_epsilon(e):
        if e == "no-diff-privacy":
            return float('inf')
        return float(e)

    # reverse=True adicionado para inverter a ordenação
    epsilons_ordenados = sorted(dados_estrategia.keys(), key=order_epsilon, reverse=True)
    
    dados_plot = []
    labels_x = []

    for eps in epsilons_ordenados:
        dados_plot.append(dados_estrategia[eps])
        labels_x.append("Sem Priv." if eps == "no-diff-privacy" else str(eps))

    plt.figure(figsize=(9, 6))
    
    plt.boxplot(dados_plot, labels=labels_x, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))

    plt.title(f"Distribuição de {metric_name.upper()} por Epsilon — {estrategia_alvo.upper()}", fontsize=14)
    plt.xlabel("Nível de Privacidade (Epsilon)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(f"{metric_name}_boxplot.pdf")
    plt.close()


def plotar_impacto_epsilon_barras(dados_agregados, estrategia_alvo, metric_name):
    """
    Gera um gráfico de barras com ordenação invertida (Sem Priv. até Epsilon menor).
    """
    if estrategia_alvo not in dados_agregados:
        print(f"Estratégia '{estrategia_alvo}' não encontrada para gráfico de barras.")
        return

    dados_estrategia = dados_agregados[estrategia_alvo]
    
    def order_epsilon(e):
        if e == "no-diff-privacy":
            return float('inf')
        return float(e)

    # reverse=True adicionado para inverter a ordenação
    epsilons_ordenados = sorted(dados_estrategia.keys(), key=order_epsilon, reverse=True)
    
    medias = []
    desvios = []
    labels_x = []

    for eps in epsilons_ordenados:
        valores = dados_estrategia[eps]
        medias.append(np.mean(valores))
        desvios.append(np.std(valores))
        labels_x.append("Sem Priv." if eps == "no-diff-privacy" else str(eps))

    plt.figure(figsize=(9, 6))
    
    plt.bar(labels_x, medias, yerr=desvios, color='skyblue', edgecolor='navy',
            capsize=6, alpha=0.85, label=f'Estratégia: {estrategia_alvo}')

    plt.title(f"Média de {metric_name.upper()} por Epsilon — {estrategia_alvo.upper()}", fontsize=14)
    plt.xlabel("Nível de Privacidade (Epsilon)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{metric_name}_barras.pdf")
    plt.close()


if __name__ == "__main__":
    caminho_base = "results"
    metrica_alvo = "final_mse"
    
    dados_processados = carregar_dados_simulacao(
        base_path=caminho_base, 
        target_metric=metrica_alvo,
        user_type="clients"
    )
    
    estrategia_selecionada = "threshold_trees" # "all_trees" 
    
    if dados_processados:
        # 1. Gráfico de Linhas (Invertido)
        plotar_impacto_epsilon(dados_processados, estrategia_alvo=estrategia_selecionada, metric_name=metrica_alvo)
        
        # 2. Gráfico Boxplot (Invertido)
        plotar_impacto_epsilon_boxplot(dados_processados, estrategia_alvo=estrategia_selecionada, metric_name=metrica_alvo)
        
        # 3. Gráfico de Barras (Invertido)
        plotar_impacto_epsilon_barras(dados_processados, estrategia_alvo=estrategia_selecionada, metric_name=metrica_alvo)
        
        print("Gráficos gerados e salvos com sucesso em formato PDF (ordenação invertida)!")