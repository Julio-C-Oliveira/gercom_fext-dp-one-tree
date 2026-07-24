import json
import numpy as np
import pulp
from scipy.optimize import linear_sum_assignment
from sklearn.tree import DecisionTreeRegressor

from fedt.app.settings import paths, dataset, settings
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client

def fit_client(X, y, epsilon, seed):
    model = DecisionTreeRegressor(
        max_depth=settings.differential_privacy.tree_max_depth,
        splitter=settings.differential_privacy.splitter,
        random_state=seed
    )
    model.fit(
        X, y,
        global_max_target=settings.differential_privacy.global_max_target,
        global_min_target=settings.differential_privacy.global_min_target,
        epsilon_global_budget=epsilon,
        balancing_coefficient=settings.differential_privacy.balancing_coefficient
    )
    return model

def extract_tree_geometry(model, n_features):
    tree = model.tree_
    geometry_leafs = []

    def travel_nodes(node_id, bounds):
        if tree.children_left[node_id] == -1:
            geometry_leafs.append({
                'bounds': bounds,
                'n_samples': tree.n_node_samples[node_id],
                'value': tree.value[node_id][0][0]
            })
            return

        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        left_bounds = [b.copy() for b in bounds]
        new_up_lim = min(left_bounds[feature_idx][1], threshold)
        left_bounds[feature_idx][1] = max(left_bounds[feature_idx][0], new_up_lim)
        travel_nodes(tree.children_left[node_id], left_bounds)

        right_bounds = [b.copy() for b in bounds]
        new_botton_lim = max(right_bounds[feature_idx][0], threshold)
        right_bounds[feature_idx][0] = min(right_bounds[feature_idx][1], new_botton_lim)
        travel_nodes(tree.children_right[node_id], right_bounds)

    initial_limits = [[0.0, 1.0] for _ in range(n_features)]
    travel_nodes(0, initial_limits)

    return geometry_leafs

def perform_data_recovery_attack(geometry_leafs, n_features):
    """
    Ataque de reconstrução baseado na geometria de UMA árvore.

    Usa programação linear (L1-norm) para minimizar a dispersão dos valores
    reconstruídos de y em torno da média de cada folha, ao invés de apenas
    satisfazer as restrições sem objetivo (prob += 0).

    A minimização da Norma L1 (soma dos desvios absolutos) é linearizada via
    variáveis auxiliares e_i que representam |y_i - mu|, com restrições:
        e_i >= y_i - mu
        e_i >= mu - y_i
    """
    prob = pulp.LpProblem("Recovery_Attack", pulp.LpMinimize)

    X_reconstructed_vars = []
    y_reconstructed_vars = []
    error_vars = []  # Desvios absolutos |y_i - mu| para L1-norm

    idx_point = 0
    for idx, leaf in enumerate(geometry_leafs):
        n_s = leaf['n_samples']
        mu = leaf['value']
        bounds = leaf['bounds']

        leaf_y_vars = []
        for i in range(n_s):
            x_vars = []
            for j in range(n_features):
                var_name = f"x_{idx_point}_{j}"
                lb = bounds[j][0]
                ub = bounds[j][1]
                # lowBound e upBound já garantem as restrições de bounds internamente
                v = pulp.LpVariable(var_name, lowBound=lb, upBound=ub, cat='Continuous')
                x_vars.append(v)

            # Consumo de energia nunca pode ser negativo
            y_name = f"y_{idx_point}"
            y_v = pulp.LpVariable(y_name, lowBound=0.0, cat='Continuous')

            # Variável auxiliar para o Erro Absoluto (L1 Norm): e_i >= |y_i - mu|
            e_name = f"e_{idx_point}"
            e_v = pulp.LpVariable(e_name, lowBound=0.0, cat='Continuous')

            # Linearização do valor absoluto
            prob += e_v >= y_v - mu
            prob += e_v >= mu - y_v

            leaf_y_vars.append(y_v)
            error_vars.append(e_v)
            X_reconstructed_vars.append(x_vars)
            y_reconstructed_vars.append(y_v)
            idx_point += 1

        # A soma de todos os y_i na folha deve ser igual a n_s * mu
        prob += pulp.lpSum(leaf_y_vars) == n_s * mu

    # Objetivo: minimizar a soma dos desvios absolutos (L1-norm)
    prob += pulp.lpSum(error_vars)

    status = prob.solve(pulp.COIN_CMD(path="/usr/bin/cbc", msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"O solver falhou com status '{pulp.LpStatus[status]}'.")

    # Fallback para None: variáveis degeneradas (lb == ub) podem retornar None
    def _safe_value(var):
        v = pulp.value(var)
        return v if v is not None else (var.lowBound or 0.0)

    X_rec = np.array([[_safe_value(x) for x in row] for row in X_reconstructed_vars])
    y_rec = np.array([_safe_value(y) for y in y_reconstructed_vars])

    return X_rec, y_rec

def _hyperrect_overlap_fraction(bounds_a, bounds_b):
    """
    Calcula a fração do volume do hiperretângulo A coberta pelo hiperretângulo B.

    Retorna um valor em [0, 1]:
      - 0.0  → sem sobreposição
      - 1.0  → B contém A completamente

    A fração é calculada como volume(A ∩ B) / volume(A) em cada dimensão.
    Dimensões degeneradas (lb == ub) são tratadas separadamente.
    """
    vol_a = 1.0
    vol_intersect = 1.0
    for (la, ua), (lb, ub) in zip(bounds_a, bounds_b):
        dim_a = max(0.0, ua - la)
        dim_intersect = max(0.0, min(ua, ub) - max(la, lb))
        if dim_a < 1e-12:
            # Dimensão degenerada: verifica se o ponto la está dentro de [lb, ub]
            if not (lb - 1e-10 <= la <= ub + 1e-10):
                return 0.0
            # Não contribui para o volume (dimensão zero)
        else:
            vol_a *= dim_a
            vol_intersect *= dim_intersect
    if vol_a < 1e-20:
        return 0.0
    return vol_intersect / vol_a


def perform_intersection_attack(all_extracted_geometries, n_features, n_samples):
    """
    Ataque de intersecção que cruza a geometria de MÚLTIPLAS árvores do ensemble.

    Por que a abordagem anterior causava Infeasible:
    As folhas de uma árvore PARTICIONAM o espaço de features (são mutuamente
    exclusivas). Intersecionando os bounds de TODAS as folhas de uma árvore
    obtemos um conjunto vazio — causando infeasibility no LP.

    Estratégia correta (LP Relaxado com Consenso Geométrico):
    1. FACTIBILIDADE: Usa a estrutura da árvore-âncora (tree 0) para criar as
       variáveis e as restrições de bounds e de soma — idêntico ao
       perform_data_recovery_attack, que é sempre feasible.
    2. CONSENSO DE μ: Para cada folha da âncora (com bounds B0 e média μ0),
       calcula um μ de consenso ponderando as contribuições das folhas das
       outras 29 árvores pela fração de sobreposição de volume:
           fração = volume(B0 ∩ Bt) / volume(B0)
       Isso usa a informação geométrica de 30 árvores para refinar a estimativa
       de y sem adicionar restrições contraditórias ao LP.
    3. OBJETIVO: Minimiza L1-norm em relação ao μ de consenso (mais preciso que μ0).
    """
    prob = pulp.LpProblem("Intersection_Recovery_Attack", pulp.LpMinimize)

    X_reconstructed_vars = []
    y_reconstructed_vars = []
    error_vars = []

    anchor_geometry = all_extracted_geometries[0]

    # Pré-computar o μ de consenso para cada folha da árvore âncora.
    # Para cada folha âncora (B0, μ0), iterar sobre as folhas de todas as
    # outras árvores e ponderar μt pela fração de sobreposição de volume
    # volume(B0 ∩ Bt) / volume(B0). Folhas com maior sobreposição contribuem
    # mais para o consenso — capturando a informação geométrica do ensemble.
    consensus_mus = []
    for anchor_leaf in anchor_geometry:
        mu_0 = anchor_leaf['value']
        bounds_0 = anchor_leaf['bounds']

        total_weight = 1.0          # Árvore âncora: peso 1
        weighted_mu_sum = mu_0      # Contribuição da árvore âncora

        for geometry_leafs in all_extracted_geometries[1:]:
            for leaf_t in geometry_leafs:
                overlap = _hyperrect_overlap_fraction(bounds_0, leaf_t['bounds'])
                if overlap > 1e-8:
                    weighted_mu_sum += overlap * leaf_t['value']
                    total_weight += overlap

        consensus_mus.append(weighted_mu_sum / total_weight)

    # Construir o LP usando a estrutura da árvore âncora (garante factibilidade)
    idx_point = 0
    for leaf_idx, anchor_leaf in enumerate(anchor_geometry):
        n_s = anchor_leaf['n_samples']
        mu_0 = anchor_leaf['value']
        bounds_0 = anchor_leaf['bounds']
        mu_consensus = consensus_mus[leaf_idx]

        leaf_y_vars = []
        for i in range(n_s):
            x_vars = []
            for j in range(n_features):
                lb = bounds_0[j][0]
                ub = bounds_0[j][1]
                # Bounds da folha âncora para x: garante que x ∈ hiperretângulo da folha
                v = pulp.LpVariable(f"x_{idx_point}_{j}", lowBound=lb, upBound=ub, cat='Continuous')
                x_vars.append(v)

            # y com lower bound 0 (consumo de energia não negativo)
            y_v = pulp.LpVariable(f"y_{idx_point}", lowBound=0.0, cat='Continuous')

            # Variável L1: desvio absoluto de y_i em relação ao μ de consenso
            # O consenso é mais informativo que μ0 pois incorpora 30 árvores
            e_v = pulp.LpVariable(f"e_{idx_point}", lowBound=0.0, cat='Continuous')
            prob += e_v >= y_v - mu_consensus
            prob += e_v >= mu_consensus - y_v
            error_vars.append(e_v)

            leaf_y_vars.append(y_v)
            X_reconstructed_vars.append(x_vars)
            y_reconstructed_vars.append(y_v)
            idx_point += 1

        # Restrição hard de soma: Σy_i = n_s * μ0 (usa μ original da folha âncora)
        # O objetivo suave usa μ_consensus; a restrição hard garante consistência
        # com a árvore treinada no dado real do cliente.
        prob += pulp.lpSum(leaf_y_vars) == n_s * mu_0

    # Objetivo: minimizar soma total dos desvios L1 em relação ao μ de consenso
    prob += pulp.lpSum(error_vars)

    status = prob.solve(pulp.COIN_CMD(path="/usr/bin/cbc", msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"O solver de intersecção falhou com status '{pulp.LpStatus[status]}'.")

    # Extrair valores com fallback: pulp pode retornar None para variáveis
    # degeneradas (lb == ub) — nesses casos, usamos o valor do bound.
    def _safe_value(var):
        v = pulp.value(var)
        return v if v is not None else (var.lowBound or 0.0)

    X_rec = np.array([[_safe_value(x) for x in row] for row in X_reconstructed_vars])
    y_rec = np.array([_safe_value(y) for y in y_reconstructed_vars])

    return X_rec, y_rec

def evaluate_attack_sucess_rate(X_real, y_real, X_rec, y_rec):
    cost_matriz = np.linalg.norm(X_real[:, None, :] - X_rec[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matriz)

    align_X_rec = X_rec[col_ind]
    align_y_rec = y_rec[col_ind]

    mse_X = np.mean((X_real - align_X_rec) ** 2)
    mse_y = np.mean((y_real - align_y_rec) ** 2)
    rmse_y = np.sqrt(mse_y)

    return mse_y, rmse_y

if __name__ == "__main__":
    client_data_divisor = 25  # De 1 a 100
    n_trees = 30              # Número de árvores do ensemble para o ataque de intersecção

    epsilon_list = [setting.epsilon for setting in simulation.epsilon_settings]

    result_dict_mse = {eps: [] for eps in epsilon_list}
    result_dict_rmse = {eps: [] for eps in epsilon_list}

    for seed in simulation.seeds:
        X_train_raw, y_train_raw, _, _ = load_house_client(
            seed=seed,
            alpha=settings.client.dirichlet_alpha,
            bins=settings.client.number_of_bins_for_dirichlet,
            percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client / client_data_divisor
        )
        X_real = X_train_raw.to_numpy()
        y_real = y_train_raw.to_numpy()

        X_min = X_real.min(axis=0)
        X_max = X_real.max(axis=0)

        X_max[X_max == X_min] += 1e-8  # Evitar divisão por zero
        X_real_norm = (X_real - X_min) / (X_max - X_min)

        n_samples, n_features = X_real_norm.shape
        print(f"Seed: {seed} | Amostras selecionadas: {n_samples} | Features: {n_features}")

        for epsilon in epsilon_list:
            try:
                # 1. Gerar as geometrias das n_trees árvores do mesmo cliente
                all_extracted_geometries = []

                for t in range(n_trees):
                    # Cada árvore usa uma sub-seed ligeiramente diferente,
                    # simulando o ensemble federated com variação de inicialização.
                    tree_seed = seed + t
                    intercepted_model = fit_client(X_real_norm, y_real, epsilon=epsilon, seed=tree_seed)
                    extracted_geometry = extract_tree_geometry(intercepted_model, n_features=n_features)
                    all_extracted_geometries.append(extracted_geometry)

                # 2. Executar o ataque de intersecção sobre as n_trees geometrias
                X_attacked, y_attacked = perform_intersection_attack(
                    all_extracted_geometries,
                    n_features=n_features,
                    n_samples=n_samples
                )

                mse_y, rmse_y = evaluate_attack_sucess_rate(X_real_norm, y_real, X_attacked, y_attacked)

                result_dict_mse[epsilon].append(mse_y)
                result_dict_rmse[epsilon].append(rmse_y)

            except Exception as e:
                print(f"[!] Erro ao executar o pipeline para Epsilon = {epsilon}, Seed = {seed}.")
                print(f"Detalhes do Erro: {e}")
                print("----------------------------------------")

    print("\n\n" + "="*60)
    print("📊 RELATÓRIO FINAL: RMSE Y MÉDIO POR NÍVEL DE PRIVACIDADE")
    print("="*60)

    for epsilon in epsilon_list:
        result_list = result_dict_rmse[epsilon]
        total_hits = len(result_list)

        if total_hits > 0:
            mean_rmse = np.mean(result_list)
            print(f"Epsilon {str(epsilon):>5} | RMSE Y Médio: {mean_rmse:12.6f} | (Sucesso em {total_hits}/{len(simulation.seeds)} seeds)")
        else:
            print(f"Epsilon {str(epsilon):>5} | RMSE Y Médio: FALHA EM TODAS AS SEEDS")

    print("="*60)

    output_dir = paths.results_folder / "side_tests" / "data_reconstruction_attack"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_to_save = {
        "mse": result_dict_mse,
        "rmse": result_dict_rmse
    }

    file_path = output_dir / "dra_results.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"[+] Dados salvos com sucesso em: {file_path}")