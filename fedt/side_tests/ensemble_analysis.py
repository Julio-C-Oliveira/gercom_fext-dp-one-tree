import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from fedt.app.settings import settings, dataset, paths
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client, load_dataset_for_server, load_server_side_validation_data, get_final_seed

import warnings
from scipy.stats import ConstantInputWarning

# Ignora o aviso de feature names do scikit-learn
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def fit_local_tree(X, y, epsilon, seed):
    """Treina o modelo local com as configurações exatas de Privacidade Diferencial."""
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

def build_global_model(strategy, epsilon_setting, base_seed, num_clients):
    """Simula o processo do servidor: treina as árvores e aplica a estratégia de agregação."""
    global_model = RandomForestRegressor(
        n_estimators=num_clients,
        max_depth=settings.differential_privacy.tree_max_depth,
        warm_start=True,
        random_state=base_seed
    )
    
    X_server, y_server = load_dataset_for_server(base_seed)
    global_model.fit(X_server, y_server)
    
    client_trees = []
    for client_id in range(num_clients):
        client_seed = get_final_seed(client_id, base_seed)
        
        X_train_raw, y_train_raw, _, _ = load_house_client(
            seed=client_seed, 
            alpha=settings.client.dirichlet_alpha, 
            bins=settings.client.number_of_bins_for_dirichlet,
            percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
        )
        
        client_tree = fit_local_tree(X_train_raw, y_train_raw, epsilon_setting.epsilon, client_seed)
        client_trees.append(client_tree)
        
    if strategy == "all_trees":
        global_model.estimators_ = client_trees
        global_model.n_estimators = len(client_trees)
        
    elif strategy == "threshold_trees":
        val_seed = get_final_seed(num_clients, base_seed)
        X_val, y_val = load_server_side_validation_data(val_seed)
        
        if epsilon_setting.threshold_type == "pearson":
            eval_function = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
        else:
            eval_function = mean_squared_error

        tree_scores = [eval_function(y_val, tree.predict(X_val)) for tree in client_trees]
        
        current_threshold = epsilon_setting.threshold_value
        selected_trees = [client_trees[i] for i in range(num_clients) if tree_scores[i] < current_threshold]
        
        while not selected_trees:
            current_threshold *= epsilon_setting.threshold_multiplier
            selected_trees = [client_trees[i] for i in range(num_clients) if tree_scores[i] < current_threshold]
            
        global_model.estimators_ = selected_trees
        global_model.n_estimators = len(selected_trees)
        
    return global_model

if __name__ == "__main__":
    seeds = simulation.seeds
    num_clients = simulation.number_of_clients_for_test
    tree_percentages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    # Dicionário serializável para o JSON (Chave composta por string)
    global_results = {}
    
    print("⏳ Iniciando simulações do Ensemble em background...")
    
    for setting in simulation.epsilon_settings:
        epsilon = setting.epsilon
        
        for strategy in simulation.aggregation_strategies:
            # Chave string única para representação no arquivo JSON
            scenario_key = f"{epsilon}__{strategy}"
            
            global_results[scenario_key] = {
                str(pct): {'r2': [], 'rmse': [], 'mse': [], 'rel_mse': []} for pct in tree_percentages
            }
            
            print(f"🔄 Executando cenário: Epsilon = {epsilon} | Estratégia = {strategy}")
            
            for seed in seeds:
                global_model = build_global_model(strategy, setting, seed, num_clients)
                todas_as_arvores = global_model.estimators_
                total_arvores = len(todas_as_arvores)
                
                val_seed = get_final_seed(num_clients, seed)
                X_val, y_val = load_server_side_validation_data(val_seed)
                
                baseline_mse = None
                
                for pct in tree_percentages:
                    n_trees = max(1, int(total_arvores * pct))
                    subset_trees = todas_as_arvores[:n_trees]
                    
                    tree_predictions = np.array([tree.predict(X_val) for tree in subset_trees])
                    y_pred_subset = np.mean(tree_predictions, axis=0)
                    
                    mse = mean_squared_error(y_val, y_pred_subset)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_val, y_pred_subset)
                    
                    if pct == 1.0:
                        baseline_mse = mse
                        rel_mse = 0.0
                    else:
                        rel_mse = ((mse - baseline_mse) / baseline_mse) * 100
                    
                    # Armazenamento estruturado
                    pct_str = str(pct)
                    global_results[scenario_key][pct_str]['r2'].append(float(r2))
                    global_results[scenario_key][pct_str]['rmse'].append(float(rmse))
                    global_results[scenario_key][pct_str]['mse'].append(float(mse))
                    global_results[scenario_key][pct_str]['rel_mse'].append(float(rel_mse))

    # Salvamento dos dados brutos consolidados
    output_dir = paths.results_folder / "side_tests" / "ensemble_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ensemble_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(global_results, f, indent=4)
        
    print(f"\n🏁 Simulação concluída! Resultados salvos em: {output_file}")