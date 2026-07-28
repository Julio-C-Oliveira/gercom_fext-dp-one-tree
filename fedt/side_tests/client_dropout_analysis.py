import json
import numpy as np
import pandas as pd  # Adicionado para facilitar a concatenação dos dados de teste
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from fedt.app.settings import settings, dataset, paths
from fedt.app.server_strategy import Strategy
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

if __name__ == "__main__":
    seeds = simulation.seeds
    num_clients = simulation.number_of_clients_for_test
    # tree_percentages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    tree_percentages = [1.0, 0.75, 0.5, 0.25, 0.1]

    global_results = {}
    
    print("⏳ Pré-computando modelos dos clientes e dados de teste...")
    
    # Dicionários para cache
    # Estrutura: { epsilon: { seed: [modelos_dos_clientes] } }
    precomputed_models = {}
    # Estrutura: { seed: (X_test_agregado, y_test_agregado) }
    precomputed_test_data = {}

    for setting in simulation.epsilon_settings:
        epsilon = setting.epsilon
        precomputed_models[epsilon] = {}

        print(f"Computando o epsilon: {epsilon}")
        
        for seed in seeds:
            client_trees = []
            X_test_list = []
            y_test_list = []
            
            for client_id in range(num_clients):
                client_seed = get_final_seed(client_id, seed)
                
                # Coletando também o X_test e y_test de cada cliente
                X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_house_client(
                    seed=client_seed, 
                    alpha=settings.client.dirichlet_alpha, 
                    bins=settings.client.number_of_bins_for_dirichlet,
                    percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
                )
                
                client_tree = fit_local_tree(X_train_raw, y_train_raw, epsilon, client_seed)
                client_trees.append(client_tree)
                
                # Armazena os dados de teste deste cliente
                X_test_list.append(X_test_raw)
                y_test_list.append(y_test_raw)

            # Salva os modelos processados no cache para este epsilon e seed
            precomputed_models[epsilon][seed] = client_trees
            
            # Como os dados de teste não dependem do epsilon, só precisamos agregar e salvar uma vez por seed
            if seed not in precomputed_test_data:
                # Concatena os dados de teste de todos os clientes em um único bloco de avaliação
                if isinstance(X_test_list[0], pd.DataFrame) or isinstance(X_test_list[0], pd.Series):
                    X_test_all = pd.concat(X_test_list, axis=0)
                    y_test_all = pd.concat(y_test_list, axis=0)
                else:
                    X_test_all = np.vstack(X_test_list)
                    y_test_all = np.concatenate(y_test_list)
                
                precomputed_test_data[seed] = (X_test_all, y_test_all)

    print("✅ Pré-computação concluída. Iniciando simulações do Ensemble em background...")
    
    for setting in simulation.epsilon_settings:
        epsilon = setting.epsilon
        
        for strategy in simulation.aggregation_strategies:
            scenario_key = f"{epsilon}____{strategy}"
            
            global_results[scenario_key] = {
                str(pct): {'r2': [], 'rmse': [], 'mse': [], 'rel_mse': []} for pct in tree_percentages
            }
            
            print(f"🔄 Executando cenário: Epsilon = {epsilon} | Estratégia = {strategy}")
            
            for seed in seeds:
                # Carrega validação do servidor (ainda necessário para as estratégias threshold)
                val_seed = get_final_seed(num_clients, seed)
                X_val, y_val = load_server_side_validation_data(val_seed)
                validation_dataset = (X_val, y_val)
                
                # Recupera os dados cacheados (Otimização O(1) sem treinar novamente)
                client_trees = precomputed_models[epsilon][seed]
                X_test_global, y_test_global = precomputed_test_data[seed]

                rng = np.random.default_rng(seed)
                baseline_mse = None
                
                for pct in tree_percentages:
                    n_trees = max(1, int(num_clients * pct))
                    
                    indices_selecionados = rng.choice(num_clients, size=n_trees, replace=False)
                    subset_received_trees = [client_trees[i] for i in indices_selecionados]
                    
                    global_model = RandomForestRegressor(
                        n_estimators=len(subset_received_trees),
                        max_depth=settings.differential_privacy.tree_max_depth,
                        warm_start=True,
                        random_state=seed
                    )
                    
                    X_server, y_server = load_dataset_for_server(seed)
                    global_model.fit(X_server, y_server)
                    
                    match strategy:
                        case "ensemble_all_trees":
                            global_model.estimators_ = Strategy.ensemble_all_trees(subset_received_trees)
                            global_model.n_estimators = len(global_model.estimators_)

                        case "ensemble_threshold_trees":
                            global_model.estimators_ = Strategy.ensemble_threshold_trees(
                                validation_dataset=validation_dataset,
                                received_trees=subset_received_trees,
                                threshold_type=setting.threshold_type,
                                threshold_value=setting.threshold_value,
                                threshold_multiplier=setting.threshold_multiplier,
                            )
                            global_model.n_estimators = len(global_model.estimators_)

                        case "merge_all_trees":
                            merged = Strategy.merge_all_trees(
                                received_trees=subset_received_trees,
                                max_depth_global=settings.differential_privacy.tree_max_depth,
                                seed=seed,
                            )
                            global_model.estimators_ = [merged]
                            global_model.n_estimators = 1

                        case "merge_threshold_trees":
                            merged = Strategy.merge_threshold_trees(
                                validation_dataset=validation_dataset,
                                received_trees=subset_received_trees,
                                threshold_type=setting.threshold_type,
                                threshold_value=setting.threshold_value,
                                threshold_multiplier=setting.threshold_multiplier,
                                max_depth_global=settings.differential_privacy.tree_max_depth,
                                seed=seed,
                            )
                            global_model.estimators_ = [merged]
                            global_model.n_estimators = 1

                        case _:
                            raise ValueError(f"Estratégia desconhecida: '{strategy}'")

                    # Avaliação sendo feita agora nos dados de teste agregados dos clientes, e não no X_val
                    y_pred_subset = global_model.predict(X_test_global)
                    
                    mse = mean_squared_error(y_test_global, y_pred_subset)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test_global, y_pred_subset)
                    
                    if pct == 1.0:
                        baseline_mse = mse
                        rel_mse = 0.0
                    else:
                        rel_mse = ((mse - baseline_mse) / baseline_mse) * 100
                    
                    pct_str = str(pct)
                    global_results[scenario_key][pct_str]['r2'].append(float(r2))
                    global_results[scenario_key][pct_str]['rmse'].append(float(rmse))
                    global_results[scenario_key][pct_str]['mse'].append(float(mse))
                    global_results[scenario_key][pct_str]['rel_mse'].append(float(rel_mse))

    output_dir = paths.results_folder / "side_tests" / "client_dropout_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "client_dropout_analysis.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(global_results, f, indent=4)
        
    print(f"\n🏁 Simulação concluída! Resultados salvos em: {output_file}")