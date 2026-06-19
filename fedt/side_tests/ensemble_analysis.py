import os
import matplotlib
matplotlib.use('Agg') # Modo headless para salvar os gráficos sem depender de interface visual
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from fedt.app.settings import settings, dataset, paths
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client, load_dataset_for_server, load_server_side_validation_data, get_final_seed

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
    """
    Simula o processo do servidor: treina as árvores e aplica a estratégia de agregação.
    """
    global_model = RandomForestRegressor(
        n_estimators=num_clients,
        max_depth=settings.differential_privacy.tree_max_depth,
        warm_start=True,
        random_state=base_seed
    )
    
    # O servidor faz um fit dummy inicial para inicializar a estrutura do RF
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
        
    # --- Aplicação da Estratégia de Agregação ---
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
    
    # Array de proporções de árvores (100% até 10%)
    tree_percentages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    # Estrutura para consolidar dados de todas as rodadas
    # Chave: (epsilon, estrategia) -> Valor: { pct: { r2: [], rmse: [], mse: [], rel_mse: [] } }
    global_results = {}
    
    print("⏳ Iniciando simulações em background...")
    
    for setting in simulation.epsilon_settings:
        epsilon = setting.epsilon
        
        for strategy in simulation.aggregation_strategies:
            scenario_key = (epsilon, strategy)
            
            # Inicializa a estrutura para este cenário específico
            global_results[scenario_key] = {
                pct: {'r2': [], 'rmse': [], 'mse': [], 'rel_mse': []} for pct in tree_percentages
            }
            
            print(f"🔄 Executando cenário: Epsilon = {epsilon} | Estratégia = {strategy} (Avaliando todas as seeds...)")
            
            for seed in seeds:
                # 1. Construir o Modelo Global
                global_model = build_global_model(strategy, setting, seed, num_clients)
                todas_as_arvores = global_model.estimators_
                total_arvores = len(todas_as_arvores)
                
                # 2. Carregar dados de validação globais
                val_seed = get_final_seed(num_clients, seed)
                X_val, y_val = load_server_side_validation_data(val_seed)
                
                baseline_mse = None
                
                # 3. Iterar sobre as porcentagens de árvores coletando métricas internamente
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
                    
                    # Armazena os dados para posterior média global
                    global_results[scenario_key][pct]['r2'].append(r2)
                    global_results[scenario_key][pct]['rmse'].append(rmse)
                    global_results[scenario_key][pct]['mse'].append(mse)
                    global_results[scenario_key][pct]['rel_mse'].append(rel_mse)

    # =========================================================================
    # 📊 IMPRESSÃO DA VISÃO GERAL GLOBAL E GERAÇÃO DE GRÁFICOS
    # =========================================================================
    print(f"\n\n{'='*85}")
    print(f"📊 VISÃO GERAL CONSOLIDADA DOS RESULTADOS (MÉDIA DE {len(seeds)} SEEDS)")
    print(f"{'='*85}")

    for scenario_key, pct_data in global_results.items():
        epsilon, strategy = scenario_key
        print(f"\n📈 Cenário: Epsilon = {epsilon} | Estratégia = {strategy}")
        print(f"{'-'*85}")
        
        # Listas para alimentar o matplotlib
        plot_pcts = []
        plot_avg_r2 = []
        plot_avg_rel_mse = []
        
        for pct in tree_percentages:
            # Consolida a média de todas as seeds para o percentual atual
            avg_r2 = np.mean(pct_data[pct]['r2'])
            avg_rmse = np.mean(pct_data[pct]['rmse'])
            avg_mse = np.mean(pct_data[pct]['mse'])
            avg_rel_mse = np.mean(pct_data[pct]['rel_mse'])
            
            plot_pcts.append(int(pct * 100))
            plot_avg_r2.append(avg_r2)
            plot_avg_rel_mse.append(avg_rel_mse)
            
            pct_str = f"{int(pct*100)}%".rjust(4)
            print(f"[{pct_str} das Árvores] "
                  f"Média R²: {avg_r2:>7.4f} | "
                  f"Média RMSE: {avg_rmse:>7.4f} | "
                  f"Média MSE: {avg_mse:>8.4f} | "
                  f"Média Var. Relativa MSE: {avg_rel_mse:>+7.2f}%")
            
        # --- GERAÇÃO DOS GRÁFICOS COM MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Degradação do Ensemble sob Privacidade Diferencial\n(Epsilon: {epsilon} | Estratégia: {strategy} | Médias baseadas em {len(seeds)} Seeds)", fontsize=12, fontweight='bold')
        
        # Subplot 1: R² Score (Capacidade de Generalização)
        ax1.plot(plot_pcts, plot_avg_r2, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Média R²')
        ax1.set_title("Evolução da Generalização (R² Score)")
        ax1.set_xlabel("% de Árvores Preservadas no Modelo Global")
        ax1.set_ylabel("R² Score (Maior é melhor)")
        ax1.set_xticks(plot_pcts)
        ax1.set_xlim(105, 5) # Força o eixo X a exibir decrescente (de 100% a 10%)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Subplot 2: Variação Relativa do MSE (Crescimento do erro)
        ax2.plot(plot_pcts, plot_avg_rel_mse, marker='s', linestyle='-', color='#d62728', linewidth=2, label='Var. Relativa MSE')
        ax2.set_title("Aumento Percentual do Erro (Var. Relativa MSE)")
        ax2.set_xlabel("% de Árvores Preservadas no Modelo Global")
        ax2.set_ylabel("Aumento do Erro em relação ao baseline de 100% (%)")
        ax2.set_xticks(plot_pcts)
        ax2.set_xlim(105, 5) # Força o eixo X a exibir decrescente (de 100% a 10%)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_dir = paths.graphics_path / "ensemble_analysis"
        filename = output_dir / f"degradacao_ensemble_eps_{epsilon}_{strategy}.pdf"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Gráfico salvo com sucesso: {filename}")
        print(f"{'-'*85}")

    print("\n🏁 Processo completo finalizado com sucesso!")