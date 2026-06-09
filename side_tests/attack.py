import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pulp
from scipy.optimize import linear_sum_assignment

# ==========================================
# CONFIGURAÇÕES DE PASSARELA (MOCK)
# ==========================================
class Paths:
    dataset_path = "energydata_complete.csv"

class DatasetConfig:
    percentage_value_of_samples_per_client = 0.20

paths = Paths()
dataset = DatasetConfig()
# ==========================================

def load_dataset():
    energy_data_complete = pd.read_csv(paths.dataset_path)
    columns_for_training = []
    temperature_columns = [f"T{i}" for i in range(1, 10)]
    humidity_columns = [f"RH_{i}" for i in range(1, 10)]

    for temperature in temperature_columns:
        columns_for_training.append(temperature)
        
    for humidity in humidity_columns:
        columns_for_training.append(humidity)
        
    columns_for_training.append("T_out")
    columns_for_training.append("RH_out")
    columns_for_training.append("Press_mm_hg")
    columns_for_training.append("Visibility")

    data = energy_data_complete[columns_for_training]
    label = energy_data_complete["Appliances"]
    
    return data, label

def load_house_client(seed, alpha, bins):
    rng = np.random.default_rng(seed)
    X, y = load_dataset()
    number_of_samples = int((len(X) * dataset.percentage_value_of_samples_per_client) / 100)

    y_bins = pd.qcut(y, q=bins, labels=False, duplicates='drop')
    num_bins = len(np.unique(y_bins))
    dirichlet_vector = rng.dirichlet([alpha] * num_bins)
    row_weights = dirichlet_vector[y_bins]
    probabilities = row_weights / row_weights.sum()
    idxs = rng.choice(X.shape[0], size=number_of_samples, replace=False, p=probabilities)
    
    X = X.iloc[idxs]
    y = y.iloc[idxs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, y_train, X_test, y_test

def treinar_modelo_cliente(X, y, epsilon, seed=42):
    print(f"[+] Etapa 1.2: Treinando a árvore de decisão com DP (Epsilon: {epsilon})...")
    modelo = DecisionTreeRegressor(
        max_depth=3,
        splitter="best",
        random_state=seed
    )
    modelo.fit(
        X, y,
        global_max_target=1200,
        global_min_target=0,
        epsilon_global_budget=epsilon,
        balancing_coefficient=0.37
    )
    return modelo

def extrair_geometria_arvore(modelo, n_features):
    print("[+] Etapa 2: Executando o parser geométrico da árvore interceptada...")
    tree = modelo.tree_
    folhas_geometria = []
    
    def percorrer_nos(node_id, bounds):
        if tree.children_left[node_id] == -1:
            folhas_geometria.append({
                'bounds': bounds,
                'n_samples': tree.n_node_samples[node_id],
                'value': tree.value[node_id][0][0]
            })
            return
        
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        bounds_esquerda = [b.copy() for b in bounds]
        novo_limite_sup = min(bounds_esquerda[feature_idx][1], threshold)
        bounds_esquerda[feature_idx][1] = max(bounds_esquerda[feature_idx][0], novo_limite_sup)
        percorrer_nos(tree.children_left[node_id], bounds_esquerda)
        
        bounds_direita = [b.copy() for b in bounds]
        novo_limite_inf = max(bounds_direita[feature_idx][0], threshold)
        bounds_direita[feature_idx][0] = min(bounds_direita[feature_idx][1], novo_limite_inf)
        percorrer_nos(tree.children_right[node_id], bounds_direita)

    limites_iniciais = [[0.0, 1.0] for _ in range(n_features)]
    percorrer_nos(0, limites_iniciais)
    
    return folhas_geometria

def executar_ataque_reconstrucao(folhas_geometria, n_features):
    print("[+] Etapa 3: Configurando o motor de otimização linear (Ataque)...")
    prob = pulp.LpProblem("Ataque_Reconstrucao_Geometrica", pulp.LpMinimize)
    
    X_reconstructed_vars = []
    y_reconstructed_vars = []
    
    ponto_idx = 0
    for idx, folha in enumerate(folhas_geometria):
        n_s = folha['n_samples']
        mu = folha['value']
        bounds = folha['bounds']
        
        folha_y_vars = []
        for i in range(n_s):
            x_vars = []
            for j in range(n_features):
                var_name = f"x_{ponto_idx}_{j}"
                lb = bounds[j][0]
                ub = bounds[j][1]
                v = pulp.LpVariable(var_name, lowBound=lb, upBound=ub, cat='Continuous')
                x_vars.append(v)
                prob += v >= lb
            
            y_name = f"y_{ponto_idx}"
            y_v = pulp.LpVariable(y_name, cat='Continuous')
            folha_y_vars.append(y_v)
            X_reconstructed_vars.append(x_vars)
            y_reconstructed_vars.append(y_v)
            ponto_idx += 1
            
        prob += pulp.lpSum(folha_y_vars) == n_s * mu
    prob += 0
    
    status = prob.solve(pulp.COIN_CMD(path="/usr/bin/cbc", msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"O solver falhou com status '{pulp.LpStatus[status]}'.")
    
    X_rec = np.array([[pulp.value(x) for x in row] for row in X_reconstructed_vars])
    y_rec = np.array([pulp.value(y) for y in y_reconstructed_vars])
    return X_rec, y_rec

def avaliar_sucesso_ataque(X_real, y_real, X_rec, y_rec):
    print("[+] Etapa 4: Avaliando métricas de vazamento e alinhamento...")
    matriz_custo = np.linalg.norm(X_real[:, None, :] - X_rec[None, :, :], axis=2)
    linha_ind, col_ind = linear_sum_assignment(matriz_custo)
    
    X_rec_alinhado = X_rec[col_ind]
    y_rec_alinhado = y_rec[col_ind]
    
    mse_X = np.mean((X_real - X_rec_alinhado) ** 2)
    mse_y = np.mean((y_real - y_rec_alinhado) ** 2)
    rmse_y = np.sqrt(mse_y)
    
    print(f"Erro Quadrático Médio de Reconstrução das Features (MSE X): {mse_X:.6f}")
    print(f"Raiz do Erro Quadrático Médio de Reconstrução do Target (RMSE Y):   {rmse_y:.6f}")
    print("----------------------------------------")
    
    return rmse_y  # <-- RETORNO ADICIONADO PARA O CÁLCULO DA MÉDIA

if __name__ == "__main__":
    print("[+] Inicializando Experimento de Múltiplas Seeds...")

    lista_epsilons = [-1.0, 10.0, 7.0, 5.0, 3.0, 1.0, 0.75, 0.5, 0.25, 0.1]
    seeds = [
        1, 2, 5, 10, 17, 20, 
        28, 37, 42, 87, 169, 
        428, 456, 632, 744, 
        766, 769, 780, 987, 
        8502, 8934, 13001, 
        191524, 191738096, 
        195864652, 308651098, 
        2249696156, 2938670577, 
        3635933258, 4130305148
    ]

    # Estrutura para armazenar todos os RMSEs obtidos para cada Epsilon
    historico_rmse_y = {eps: [] for eps in lista_epsilons}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[*] INICIANDO BATERIA PARA A SEED = {seed}")
        print(f"{'='*60}")
        
        # 1. Carregamento movido para DENTRO do loop para testar amostragens diferentes
        X_train_raw, y_train_raw, _, _ = load_house_client(seed=seed, alpha=0.5, bins=10)
        X_real = X_train_raw.to_numpy()
        y_real = y_train_raw.to_numpy()
        
        X_min = X_real.min(axis=0)
        X_max = X_real.max(axis=0)
        X_max[X_max == X_min] += 1e-8 
        X_real_norm = (X_real - X_min) / (X_max - X_min)
        
        n_samples, n_features = X_real_norm.shape
        print(f"[i] Amostras selecionadas: {n_samples} | Features: {n_features}")

        for epsilon in lista_epsilons:
            print(f"\n--- Epsilon: {epsilon} | Seed: {seed} ---")
            
            try:
                # 2. Correção: Utilizando a seed atual da iteração
                modelo_interceptado = treinar_modelo_cliente(X_real_norm, y_real, epsilon=epsilon, seed=seed)
                
                geometria_extraida = extrair_geometria_arvore(modelo_interceptado, n_features=n_features)
                X_atacado, y_atacado = executar_ataque_reconstrucao(geometria_extraida, n_features=n_features)
                
                # Coletando o RMSE gerado nesta rodada
                rmse_y = avaliar_sucesso_ataque(X_real_norm, y_real, X_atacado, y_atacado)
                historico_rmse_y[epsilon].append(rmse_y)
                
            except Exception as e:
                print(f"[!] Erro ao executar o pipeline para Epsilon = {epsilon}, Seed = {seed}.")
                print(f"Detalhes do Erro: {e}")
                print("----------------------------------------")


    # ==========================================
    # EXIBIÇÃO DO RELATÓRIO FINAL DE MÉDIAS
    # ==========================================
    print("\n\n" + "="*60)
    print("📊 RELATÓRIO FINAL: RMSE Y MÉDIO POR NÍVEL DE PRIVACIDADE")
    print("="*60)
    
    for epsilon in lista_epsilons:
        lista_resultados = historico_rmse_y[epsilon]
        total_sucessos = len(lista_resultados)
        
        if total_sucessos > 0:
            media_rmse = np.mean(lista_resultados)
            print(f"Epsilon {str(epsilon):>5} | RMSE Y Médio: {media_rmse:12.6f} | (Sucesso em {total_sucessos}/{len(seeds)} seeds)")
        else:
            print(f"Epsilon {str(epsilon):>5} | RMSE Y Médio: FALHA EM TODAS AS SEEDS")
            
    print("="*60)