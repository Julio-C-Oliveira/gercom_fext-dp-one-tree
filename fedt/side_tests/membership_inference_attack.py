import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# ETAPA EXTRA: GERENCIAMENTO DE OUTLIERS E PLOTAGEM
# =====================================================================
def outliers_manager(remove_outliers, data_dict):
    if not remove_outliers:
        return data_dict

    cleaned_dict = {}
    for epsilon, group_values in data_dict.items():
        if len(group_values) >= 4:
            q1, q3 = np.percentile(group_values, [25, 75])
            interquartile_range = q3 - q1

            lim_inf_mod, lim_sup_mod = q1 - 1.5 * interquartile_range, q3 + 1.5 * interquartile_range
            lim_inf_ext, lim_sup_ext = q1 - 3.0 * interquartile_range, q3 + 3.0 * interquartile_range

            clean_data = []
            for value in group_values:
                is_extremo = value < lim_inf_ext or value > lim_sup_ext
                is_moderado = (value < lim_inf_mod or value > lim_sup_mod) and not is_extremo
                
                if remove_outliers == 'ambos' and (is_extremo or is_moderado): 
                    continue
                if remove_outliers == 'extremos' and is_extremo: 
                    continue
                if remove_outliers == 'moderados' and is_moderado: 
                    continue
                
                clean_data.append(value)
            cleaned_dict[epsilon] = clean_data
        else:
            cleaned_dict[epsilon] = list(group_values)
    return cleaned_dict

def boxplot(result_dict, file_name, y_label):
    data_plot = []
    labels = []

    for epsilon, values in result_dict.items():
        if len(values) > 0:
            data_plot.append(values)
            if epsilon == -1.0:
                labels.append("No Diff Priv")
            else:
                labels.append(str(epsilon))

    if not data_plot:
        print("[!] Não há dados válidos para plotar o gráfico.")
        return

    plt.figure(figsize=tuple(graphics.normal_figsize))
    plt.boxplot(
        data_plot, 
        labels=labels, 
        patch_artist=True, 
        boxprops=dict(facecolor='lightblue', color='blue'), 
        medianprops=dict(color='red', linewidth=2)
    )
    
    plt.xlabel("Privacy Level (ε)", fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    plt.ylabel(y_label, fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha, axis='y')
    plt.tight_layout()
    
    plt.savefig(f"{paths.graphics_path}/{file_name}")
    plt.close()

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
    # IMPRESSÃO DO RELATÓRIO FINAL E GERAÇÃO DOS GRÁFICOS
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

    # Filtragem de Outliers e Salvamento dos PDFs de Boxplot
    opcao_filtragem = 'ambos' 
    result_dict_accuracy = outliers_manager(opcao_filtragem, result_dict_accuracy)
    result_dict_auc = outliers_manager(opcao_filtragem, result_dict_auc)

    boxplot(
        result_dict_accuracy,
        file_name="mia_attack_Accuracy.pdf",
        y_label="MIA Attack Accuracy"
    )
    boxplot(
        result_dict_auc,
        file_name="mia_attack_AUC_ROC.pdf",
        y_label="MIA Attack AUC-ROC"
    )
    print("[+] Gráficos salvos com sucesso no diretório de saídas.")