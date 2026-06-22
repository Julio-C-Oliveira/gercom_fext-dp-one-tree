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
        max_depth=3,
        splitter="best",
        random_state=seed
    )
    model.fit(
        X, y,
        global_max_target=1200,
        global_min_target=0,
        epsilon_global_budget=epsilon,
        balancing_coefficient=0.37
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
    prob = pulp.LpProblem("Recovery_Attack", pulp.LpMinimize)

    X_reconstructed_vars = []
    y_reconstructed_vars = []

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
                v = pulp.LpVariable(var_name, lowBound=lb, upBound=ub, cat='Continuous')
                x_vars.append(v)
                prob += v >= lb

            y_name = f"y_{idx_point}"
            y_v = pulp.LpVariable(y_name, cat='Continuous')
            leaf_y_vars.append(y_v)
            X_reconstructed_vars.append(x_vars)
            y_reconstructed_vars.append(y_v)
            idx_point += 1

        prob += pulp.lpSum(leaf_y_vars) == n_s * mu

    prob += 0

    status = prob.solve(pulp.COIN_CMD(path="/usr/bin/cbc", msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"O solver falhou com status '{pulp.LpStatus[status]}'.")
    
    X_rec = np.array([[pulp.value(x) for x in row] for row in X_reconstructed_vars])
    y_rec = np.array([pulp.value(y) for y in y_reconstructed_vars])
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
    client_data_divisor = 25 # De 1 a 100

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
        
        X_max[X_max == X_min] += 1e-8 # Para evitar divisão por 0
        X_real_norm = (X_real - X_min) / (X_max - X_min)
        
        n_samples, n_features = X_real_norm.shape
        print(f"Seed: {seed} | Amostras selecionadas: {n_samples} | Features: {n_features}")

        for epsilon in epsilon_list:
            try:
                intercepted_model = fit_client(X_real_norm, y_real, epsilon=epsilon, seed=seed)
                
                extracted_geometry = extract_tree_geometry(intercepted_model, n_features=n_features)
                X_attacked, y_attacked = perform_data_recovery_attack(extracted_geometry, n_features=n_features)
                
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