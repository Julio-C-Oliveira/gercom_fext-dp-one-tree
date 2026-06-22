import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from fedt.app.settings import paths, dataset, settings
from fedt.scripts.settings import graphics
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

# =====================================================================
# ETAPA EXTRA: ENGENHARIA DE CARACTERÍSTICAS PARA REGRESSÃO
# =====================================================================
def extract_regression_attack_features(model, X, y_true):
    """
    Extrai métricas estruturais e de erro de um modelo de REGRESSÃO.
    """
    tree_ = model.tree_
    leaf_ids = model.apply(X)
    
    # 1. Profundidade do nó de decisão
    decision_paths = model.decision_path(X)
    depths = np.array(decision_paths.sum(axis=1)).flatten() - 1 
    
    # 2. Número de amostras na folha correspondente
    n_samples = tree_.n_node_samples[leaf_ids]
    
    # 3. Impureza da folha (MSE do Nó)
    node_mse = tree_.impurity[leaf_ids]
    
    # 4. Valor predito
    preds = model.predict(X)
    
    # 5. Erro Residual Absoluto (Métrica chave)
    residuals = np.abs(y_true - preds)
    
    return np.column_stack([depths, n_samples, node_mse, preds, residuals])

# =====================================================================
# PIPELINE PRINCIPAL DE EXECUÇÃO (MÚLTIPLAS SEEDS E EPSILONS)
# =====================================================================
if __name__ == "__main__":
    client_data_divisor = 25 
    epsilon_list = [setting.epsilon for setting in simulation.epsilon_settings]

    # Inicialização dos dicionários de resultados
    result_dict_accuracy = {eps: [] for eps in epsilon_list}
    result_dict_auc = {eps: [] for eps in epsilon_list}

    print_section("Iniciando Experimentos Multi-Seed e Multi-Epsilon para MIA")

    for seed in simulation.seeds:
        # Carregamento dos dados reais da Seed atual
        X_train_raw, y_train_raw, _, _ = load_house_client(
            seed=seed, 
            alpha=settings.client.dirichlet_alpha, 
            bins=settings.client.number_of_bins_for_dirichlet,
            percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client / client_data_divisor
        )
        X_real = X_train_raw.to_numpy()
        y_real = y_train_raw.to_numpy()
        
        # Normalização MinMax
        X_min = X_real.min(axis=0)
        X_max = X_real.max(axis=0)
        X_max[X_max == X_min] += 1e-8 
        X_real_norm = (X_real - X_min) / (X_max - X_min)
        
        n_samples, n_features = X_real_norm.shape
        print(f"\n[Seed: {seed}] Dados carregados. Amostras: {n_samples} | Features: {n_features}")

        # Divisão Cliente (Target) vs Atacante (Shadow)
        X_target, X_shadow, y_target, y_shadow = train_test_split(
            X_real_norm, y_real, test_size=0.5, random_state=seed
        )

        # Divisões internas (Membros vs Não-Membros)
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target, test_size=0.5, random_state=seed
        )
        X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
            X_shadow, y_shadow, test_size=0.5, random_state=seed
        )

        # Loop pelos Budgets de Privacidade (Epsilons)
        for epsilon in epsilon_list:
            try:
                # 1. Treinamento do Modelo Alvo (Cliente)
                target_model = DecisionTreeRegressor(
                    max_depth=settings.differential_privacy.tree_max_depth, 
                    splitter="best", 
                    random_state=seed
                )
                target_model.fit(
                    X_target_train, y_target_train,
                    global_max_target=1200,
                    global_min_target=0,
                    epsilon_global_budget=epsilon,
                    balancing_coefficient=0.37
                )

                # 2. Treinamento do Modelo Sombra (Atacante) sob as mesmas condições do alvo
                shadow_model = DecisionTreeRegressor(
                    max_depth=settings.differential_privacy.tree_max_depth, 
                    splitter="best", 
                    random_state=seed
                )
                shadow_model.fit(
                    X_shadow_train, y_shadow_train,
                    global_max_target=1200,
                    global_min_target=0,
                    epsilon_global_budget=epsilon,
                    balancing_coefficient=0.37
                )

                # 3. Engenharia de Features no Ambiente Sombra (Treino do MIA)
                attack_features_train = extract_regression_attack_features(shadow_model, X_shadow_train, y_shadow_train)
                attack_labels_train = np.ones(len(X_shadow_train)) 

                attack_features_test = extract_regression_attack_features(shadow_model, X_shadow_test, y_shadow_test)
                attack_labels_test = np.zeros(len(X_shadow_test))  

                Attack_X = np.vstack((attack_features_train, attack_features_test))
                Attack_y = np.concatenate((attack_labels_train, attack_labels_test))

                # 4. Treinar o Classificador Meta-Atacante
                attack_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
                attack_model.fit(Attack_X, Attack_y)

                # 5. Avaliação do Ataque contra o Modelo do Cliente Real (Target)
                eval_features_members = extract_regression_attack_features(target_model, X_target_train, y_target_train)
                eval_labels_members = np.ones(len(X_target_train))

                eval_features_non_members = extract_regression_attack_features(target_model, X_target_test, y_target_test)
                eval_labels_non_members = np.zeros(len(X_target_test))

                Eval_X = np.vstack((eval_features_members, eval_features_non_members))
                Eval_y = np.concatenate((eval_labels_members, eval_labels_non_members))

                mia_preds = attack_model.predict(Eval_X)
                mia_probs = attack_model.predict_proba(Eval_X)[:, 1]

                # Coleta das Métricas
                accuracy = accuracy_score(Eval_y, mia_preds)
                auc_roc = roc_auc_score(Eval_y, mia_probs)

                result_dict_accuracy[epsilon].append(accuracy)
                result_dict_auc[epsilon].append(auc_roc)
                
            except Exception as e:
                print(f"[!] Falha em Epsilon = {epsilon}, Seed = {seed}. Erro: {e}")

    # =====================================================================
    # IMPRESSÃO DO RELATÓRIO FINAL E SALVAMENTO DOS DADOS EM JSON
    # =====================================================================
    print_section("📊 RELATÓRIO FINAL: SUCESSO DO MIA POR NÍVEL DE PRIVACIDADE")
    
    for epsilon in epsilon_list:
        acc_list = result_dict_accuracy[epsilon]
        auc_list = result_dict_auc[epsilon]
        total_hits = len(acc_list)
        
        if total_hits > 0:
            print(f"Epsilon {str(epsilon):>5} | Acurácia Média: {np.mean(acc_list)*100:6.2f}% | AUC-ROC Médio: {np.mean(auc_list):.4f} | ({total_hits} seeds)")
        else:
            print(f"Epsilon {str(epsilon):>5} | SEM DADOS (FALHA EM TODAS AS SEEDS)")
            
    print("="*60)

    output_dir = paths.results_folder / "side_tests" / "membership_inference_attack"
        
    output_dir.mkdir(parents=True, exist_ok=True)

    data_to_save = {
        "accuracy": result_dict_accuracy,
        "auc": result_dict_auc
    }

    file_path = output_dir / "mia_results.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"[+] Dados salvos com sucesso em: {file_path}")