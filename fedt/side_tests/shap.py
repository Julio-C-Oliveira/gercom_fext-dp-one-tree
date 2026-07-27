import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from fedt.app.settings import settings, paths, dataset
from fedt.app.server_strategy import Strategy
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client, load_dataset_for_server, load_server_side_validation_data, get_final_seed

def fit_local_tree(X, y, epsilon, seed):
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
    """Simula o processo do servidor para as 4 estratégias de agregação."""
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

    val_seed = get_final_seed(num_clients, base_seed)
    validation_dataset = load_server_side_validation_data(val_seed)

    match strategy:
        case "ensemble_all_trees":
            global_model.estimators_ = Strategy.ensemble_all_trees(client_trees)
            global_model.n_estimators = len(global_model.estimators_)

        case "ensemble_threshold_trees":
            global_model.estimators_ = Strategy.ensemble_threshold_trees(
                validation_dataset=validation_dataset,
                received_trees=client_trees,
                threshold_type=epsilon_setting.threshold_type,
                threshold_value=epsilon_setting.threshold_value,
                threshold_multiplier=epsilon_setting.threshold_multiplier,
            )
            global_model.n_estimators = len(global_model.estimators_)

        case "merge_all_trees":
            merged = Strategy.merge_all_trees(
                received_trees=client_trees,
                max_depth_global=settings.differential_privacy.tree_max_depth,
                seed=base_seed,
            )
            global_model.estimators_ = [merged]
            global_model.n_estimators = 1

        case "merge_threshold_trees":
            merged = Strategy.merge_threshold_trees(
                validation_dataset=validation_dataset,
                received_trees=client_trees,
                threshold_type=epsilon_setting.threshold_type,
                threshold_value=epsilon_setting.threshold_value,
                threshold_multiplier=epsilon_setting.threshold_multiplier,
                max_depth_global=settings.differential_privacy.tree_max_depth,
                seed=base_seed,
            )
            global_model.estimators_ = [merged]
            global_model.n_estimators = 1

        case _:
            raise ValueError(f"Estratégia desconhecida: '{strategy}'")

    return global_model

def export_shap_data(model, X_test, output_path):
    """Calcula os valores SHAP e os serializa em JSON."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    base_value = explainer.expected_value
    if isinstance(base_value, (np.ndarray, list)):
        base_value = base_value[0]

    data = {
        "shap_values": shap_values.tolist(),
        "expected_value": float(base_value),
        "X_test": X_test.values.tolist() if isinstance(X_test, pd.DataFrame) else X_test.tolist(),
        "feature_names": X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

if __name__ == "__main__":
    results_base = paths.results_folder / "side_tests" / "shap_analysis"
    target_client_id = 0
    
    seeds = simulation.seeds
    num_clients = simulation.number_of_clients_for_test
    
    for setting in simulation.epsilon_settings:
        epsilon = setting.epsilon
        
        for strategy in simulation.aggregation_strategies:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"🔄 Epsilon: {epsilon} | Estratégia: {strategy} | Seed: {seed}")
                print(f"{'='*60}")
                
                output_dir = results_base / strategy / f"eps_{epsilon}" / f"seed_{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                client_seed = get_final_seed(target_client_id, seed)
                X_train_target, y_train_target, X_test_target, _ = load_house_client(
                    seed=client_seed, 
                    alpha=settings.client.dirichlet_alpha, 
                    bins=settings.client.number_of_bins_for_dirichlet,
                    percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
                )
                
                print("🌲 Treinando Modelo Local...")
                local_model = fit_local_tree(X_train_target, y_train_target, epsilon, client_seed)
                
                print(f"🌐 Construindo Modelo Global (Agregação: {strategy})...")
                global_model = build_global_model(strategy, setting, seed, num_clients)
                
                print("💾 Exportando matrizes SHAP para JSON...")
                export_shap_data(local_model, X_test_target, output_dir / "shap_local.json")
                export_shap_data(global_model, X_test_target, output_dir / "shap_global.json")
                
                print(f"✅ Dados exportados para: {output_dir}")