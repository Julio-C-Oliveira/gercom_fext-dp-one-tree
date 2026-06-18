import os
import matplotlib
matplotlib.use('Agg') # Modo headless para não travar interfaces gráficas
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from fedt.app.settings import settings, paths, dataset
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
        n_estimators=num_clients, # Pode diminuir se o threshold_trees descartar árvores
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
        
    elif strategy == "threshold_trees":
        # Carrega o dataset de validação do servidor com a mesma lógica do server.py
        val_seed = get_final_seed(num_clients, base_seed)
        X_val, y_val = load_server_side_validation_data(val_seed)
        
        # Define a função de avaliação
        if epsilon_setting.threshold_type == "pearson":
            eval_function = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0] # Pegamos apenas o coeficiente
        else:
            eval_function = mean_squared_error

        # Calcula o score de cada árvore
        tree_scores = [eval_function(y_val, tree.predict(X_val)) for tree in client_trees]
        
        current_threshold = epsilon_setting.threshold_value
        selected_trees = [client_trees[i] for i in range(num_clients) if tree_scores[i] < current_threshold]
        
        # Malha de segurança do threshold iterativo
        while not selected_trees:
            current_threshold *= epsilon_setting.threshold_multiplier
            selected_trees = [client_trees[i] for i in range(num_clients) if tree_scores[i] < current_threshold]
            
        global_model.estimators_ = selected_trees
        
        # Importante: atualizar o número de estimadores para refletir o corte
        global_model.n_estimators = len(selected_trees)
        
    return global_model

def generate_shap_plots(model, X_test, output_dir, prefix):
    """Gera e salva todos os gráficos SHAP."""
    print(f"  [{prefix}] Iniciando SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    base_value = explainer.expected_value
    if isinstance(base_value, (np.ndarray, list)):
        base_value = base_value[0]
        
    # --- 1. Summary Plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_dir / f"{prefix}_summary_global.pdf", bbox_inches='tight')
    plt.close()

    # --- 2. Force Plot (Para a primeira amostra) ---
    shap.force_plot(base_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.savefig(output_dir / f"{prefix}_force_plot.pdf", bbox_inches='tight')
    plt.close()

    # --- 3. Dependence Plots ---
    for col in ["T_out", "RH_out"]:
        if col in X_test.columns:
            plt.figure()
            shap.dependence_plot(col, shap_values, X_test, show=False)
            plt.savefig(output_dir / f"{prefix}_dependence_{col.lower()}.pdf", bbox_inches='tight')
            plt.close()

    # --- 4. Decision Plot ---
    plt.figure()
    shap.decision_plot(base_value, shap_values, X_test, show=False, ignore_warnings=True)
    plt.savefig(output_dir / f"{prefix}_decision_plot.pdf", bbox_inches='tight')
    plt.close()

    # --- 5. Waterfall Plot ---
    exp = shap.Explanation(
        values=shap_values[0], 
        base_values=base_value, 
        data=X_test.iloc[0], 
        feature_names=X_test.columns.tolist()
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.savefig(output_dir / f"{prefix}_waterfall_single.pdf", bbox_inches='tight')
    plt.close()

    # --- 6. Heatmap ---
    exp_all = shap.Explanation(
        values=shap_values, 
        base_values=base_value, 
        data=X_test, 
        feature_names=X_test.columns.tolist()
    )
    plt.figure()
    shap.plots.heatmap(exp_all[:100], show=False)
    plt.savefig(output_dir / f"{prefix}_heatmap.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_output_path = paths.graphics_path / "shap_analysis"
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
                
                output_dir = base_output_path / strategy / f"eps_{epsilon}" / f"seed_{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Recuperar os dados do Cliente Alvo
                client_seed = get_final_seed(target_client_id, seed)
                X_train_target, y_train_target, X_test_target, _ = load_house_client(
                    seed=client_seed, 
                    alpha=settings.client.dirichlet_alpha, 
                    bins=settings.client.number_of_bins_for_dirichlet,
                    percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
                )
                
                # 2. Treinar o Modelo Local (Apenas a visão deste cliente)
                print("🌲 Treinando Modelo Local...")
                local_model = fit_local_tree(X_train_target, y_train_target, epsilon, client_seed)
                
                # 3. Construir o Modelo Global usando a estratégia atual
                print(f"🌐 Construindo Modelo Global (Agregação: {strategy})...")
                global_model = build_global_model(strategy, setting, seed, num_clients)
                
                # 4. Avaliar Explicabilidade (SHAP)
                print("📊 Gerando Gráficos SHAP (Local vs Global)...")
                
                # Importante: Como o modelo local não muda com a estratégia do servidor (ele é gerado antes do envio), 
                # gerar o gráfico local uma vez por epsilon/seed já seria suficiente. Mas para manter
                # os pares (Local/Global) juntos na mesma pasta de resultados, geramos ambos aqui.
                generate_shap_plots(local_model, X_test_target, output_dir, prefix="LOCAL")
                generate_shap_plots(global_model, X_test_target, output_dir, prefix="GLOBAL")
                
                print(f"✅ Artefatos salvos em: {output_dir}")