import json
import logging
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau
from scipy.integrate import trapezoid

from fedt.app.settings import settings, paths, dataset
from fedt.app.server_strategy import Strategy
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client, load_dataset_for_server, load_server_side_validation_data, get_final_seed

# Parâmetro configurável para amostragem na Sensibilidade Local
SAMPLE_SIZE_LOCAL_SENSITIVITY = 100
NOISE_SCALE_LOCAL_SENSITIVITY = 0.01

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EXPLAINABILITY_EVAL")


# ==============================================================================
# 1. MODEL FIT & BUILD HELPERS
# ==============================================================================

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


# ==============================================================================
# 2. METRIC CALCULATION FUNCTIONS
# ==============================================================================

def compute_complexity_metrics(model):
    """Calcula profundidade e contagem de nós do modelo (local ou global)."""
    if isinstance(model, DecisionTreeRegressor):
        return {
            "num_trees": 1,
            "avg_depth": float(model.get_depth()),
            "max_depth": int(model.get_depth()),
            "total_nodes": int(model.tree_.node_count),
            "avg_nodes": float(model.tree_.node_count),
        }
    elif hasattr(model, "estimators_"):
        estimators = model.estimators_
        if len(estimators) == 0:
            return {"num_trees": 0, "avg_depth": 0.0, "max_depth": 0, "total_nodes": 0, "avg_nodes": 0.0}
        
        depths = [t.get_depth() for t in estimators]
        nodes = [t.tree_.node_count for t in estimators]
        return {
            "num_trees": len(estimators),
            "avg_depth": float(np.mean(depths)),
            "max_depth": int(np.max(depths)),
            "total_nodes": int(np.sum(nodes)),
            "avg_nodes": float(np.mean(nodes)),
        }
    else:
        return {"num_trees": 0, "avg_depth": 0.0, "max_depth": 0, "total_nodes": 0, "avg_nodes": 0.0}


def compute_sparsity_metrics(shap_matrix):
    """
    Mede a esparsidade / complexidade de atribuição do SHAP.
    Calcula: Hoyer Sparsity, Gini Index, Top-3 Mass, Top-5 Mass e Entropia.
    """
    N, d = shap_matrix.shape
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    total_mass = np.sum(mean_abs_shap)

    if total_mass == 0 or np.isnan(total_mass):
        return {
            "hoyer_sparsity": 0.0,
            "gini_index": 0.0,
            "top3_mass": 0.0,
            "top5_mass": 0.0,
            "attribution_entropy": 0.0
        }

    # 1. Hoyer Sparsity
    l1_norm = total_mass
    l2_norm = np.sqrt(np.sum(mean_abs_shap ** 2))
    if l2_norm > 0 and d > 1:
        hoyer = (np.sqrt(d) - (l1_norm / l2_norm)) / (np.sqrt(d) - 1.0)
        hoyer = float(np.clip(hoyer, 0.0, 1.0))
    else:
        hoyer = 0.0

    # 2. Gini Index
    diff_matrix = np.abs(mean_abs_shap[:, None] - mean_abs_shap[None, :])
    gini = float(np.sum(diff_matrix) / (2 * d * total_mass))

    # 3. Top-K Mass
    sorted_importances = np.sort(mean_abs_shap)[::-1]
    top3_mass = float(np.sum(sorted_importances[:3]) / total_mass)
    top5_mass = float(np.sum(sorted_importances[:min(5, d)]) / total_mass)

    # 4. Normalized Attribution Entropy
    prob_dist = mean_abs_shap / total_mass
    entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    norm_entropy = float(entropy / np.log(d)) if d > 1 else 0.0

    return {
        "hoyer_sparsity": hoyer,
        "gini_index": gini,
        "top3_mass": top3_mass,
        "top5_mass": top5_mass,
        "attribution_entropy": norm_entropy
    }


def compute_fidelity_metrics(model_eval, model_base, X_test):
    """Calcula concordância de predição entre modelo sob avaliação e baseline sem DP."""
    X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred_eval = model_eval.predict(X_arr)
    y_pred_base = model_base.predict(X_arr)

    mae = float(mean_absolute_error(y_pred_base, y_pred_eval))
    mse = float(mean_squared_error(y_pred_base, y_pred_eval))
    r2 = float(r2_score(y_pred_base, y_pred_eval))

    corr_res = spearmanr(y_pred_base, y_pred_eval)
    spearman_corr = float(corr_res.statistic) if not np.isnan(corr_res.statistic) else 0.0

    return {
        "r2_fidelity": r2,
        "mae_fidelity": mae,
        "mse_fidelity": mse,
        "spearman_pred_corr": spearman_corr
    }


def compute_deletion_auc(model, shap_matrix, X_test):
    """
    Calcula a curva de ablação (Deletion AUC) substituindo as features mais importantes
    pela mediana do conjunto de teste e medindo o desvio da predição.
    """
    X_arr = X_test.values.copy() if isinstance(X_test, pd.DataFrame) else X_test.copy()
    N, d = X_arr.shape
    
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    sorted_features = np.argsort(-mean_abs_shap)

    y_orig = model.predict(X_arr)
    medians = np.median(X_arr, axis=0)

    errors = [0.0]
    X_curr = X_arr.copy()

    for k in range(d):
        feat_idx = sorted_features[k]
        X_curr[:, feat_idx] = medians[feat_idx]
        y_ablated = model.predict(X_curr)
        step_error = float(np.mean(np.abs(y_orig - y_ablated)))
        errors.append(step_error)

    # AUC normalizado entre [0, 1] pelo número de passos d
    auc = float(trapezoid(errors, dx=1.0 / d))
    return auc


def compute_rank_stability(shap_base, shap_eval):
    """Calcula a estabilidade de ranking (Jaccard Top-K, Spearman, Kendall Tau) vs baseline."""
    mean_base = np.mean(np.abs(shap_base), axis=0)
    mean_eval = np.mean(np.abs(shap_eval), axis=0)

    rank_base = np.argsort(-mean_base)
    rank_eval = np.argsort(-mean_eval)

    # Jaccard Top-3
    top3_b, top3_e = set(rank_base[:3]), set(rank_eval[:3])
    jaccard_top3 = float(len(top3_b.intersection(top3_e)) / len(top3_b.union(top3_e)))

    # Jaccard Top-5
    top5_b, top5_e = set(rank_base[:5]), set(rank_eval[:5])
    jaccard_top5 = float(len(top5_b.intersection(top5_e)) / len(top5_b.union(top5_e)))

    # Spearman Rank Correlation
    sp_res = spearmanr(mean_base, mean_eval)
    spearman_rho = float(sp_res.statistic) if not np.isnan(sp_res.statistic) else 0.0

    # Kendall Tau
    kt_res = kendalltau(mean_base, mean_eval)
    kendall_tau = float(kt_res.statistic) if not np.isnan(kt_res.statistic) else 0.0

    return {
        "jaccard_top3": jaccard_top3,
        "jaccard_top5": jaccard_top5,
        "spearman_rank_corr": spearman_rho,
        "kendall_tau": kendall_tau
    }


def compute_attribution_distance(shap_base, shap_eval):
    """Calcula distâncias de atribuição SHAP (Cosine, Euclidean, Manhattan) amostra a amostra."""
    N, d = shap_base.shape

    # Euclidean per sample
    euc_dists = np.linalg.norm(shap_base - shap_eval, axis=1)
    avg_euclidean = float(np.mean(euc_dists))

    # Manhattan per sample
    man_dists = np.sum(np.abs(shap_base - shap_eval), axis=1)
    avg_manhattan = float(np.mean(man_dists))

    # Cosine Distance per sample
    dot_products = np.sum(shap_base * shap_eval, axis=1)
    norm_base = np.linalg.norm(shap_base, axis=1)
    norm_eval = np.linalg.norm(shap_eval, axis=1)
    denom = np.maximum(norm_base * norm_eval, 1e-12)
    cosine_sim = dot_products / denom
    cosine_dists = 1.0 - np.clip(cosine_sim, -1.0, 1.0)
    avg_cosine = float(np.mean(cosine_dists))

    return {
        "cosine_distance": avg_cosine,
        "euclidean_distance": avg_euclidean,
        "manhattan_distance": avg_manhattan
    }


def compute_local_sensitivity(model, X_test, shap_matrix, sample_size=SAMPLE_SIZE_LOCAL_SENSITIVITY, noise_scale=NOISE_SCALE_LOCAL_SENSITIVITY):
    """
    Adiciona ruído gaussiano minúsculo nas instâncias de teste e mede a variação dos SHAP values
    (Estimativa de Sensibilidade Local / Constante de Lipschitz).
    """
    X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    N, d = X_arr.shape

    n_samples = min(sample_size, N)
    sub_indices = np.arange(n_samples)
    
    explainer = shap.TreeExplainer(model)
    stds = np.std(X_arr, axis=0)
    stds = np.where(stds == 0, 1e-6, stds)

    sensitivities = []

    for idx in sub_indices:
        x_orig = X_arr[idx:idx+1]
        noise = np.random.normal(0, noise_scale * stds, size=(1, d))
        x_pert = x_orig + noise

        shap_orig = shap_matrix[idx:idx+1]
        shap_pert = explainer.shap_values(x_pert)

        diff_shap = np.linalg.norm(shap_orig - shap_pert)
        diff_x = np.linalg.norm(noise)

        sens = diff_shap / max(diff_x, 1e-12)
        sensitivities.append(sens)

    return float(np.mean(sensitivities))


def compute_explanation_gap(shap_local, shap_global):
    """
    Calcula a discrepância (Gap) entre a explicabilidade do Modelo Local do Cliente
    e do Modelo Global do Servidor no mesmo conjunto de dados de teste.
    """
    rank_metrics = compute_rank_stability(shap_local, shap_global)
    dist_metrics = compute_attribution_distance(shap_local, shap_global)

    return {
        "gap_cosine_distance": dist_metrics["cosine_distance"],
        "gap_euclidean_distance": dist_metrics["euclidean_distance"],
        "gap_manhattan_distance": dist_metrics["manhattan_distance"],
        "gap_jaccard_top3": rank_metrics["jaccard_top3"],
        "gap_jaccard_top5": rank_metrics["jaccard_top5"],
        "gap_spearman_rank_corr": rank_metrics["spearman_rank_corr"],
    }


# ==============================================================================
# 3. MAIN EVALUATION SIMULATION LOOP
# ==============================================================================

def run_explainability_eval():
    results_base = paths.results_folder / "side_tests" / "explainability_eval"
    results_base.mkdir(parents=True, exist_ok=True)
    
    target_client_id = 0
    seeds = simulation.seeds
    num_clients = simulation.number_of_clients_for_test
    epsilon_settings = simulation.epsilon_settings

    # Localizar a configuração baseline sem ruído (epsilon = -1.0)
    baseline_settings = [s for s in epsilon_settings if s.epsilon == -1.0]
    if len(baseline_settings) == 0:
        raise ValueError("Configuração com epsilon = -1.0 não encontrada em simulation.epsilon_settings!")
    baseline_setting = baseline_settings[0]

    all_summary_rows = []
    detailed_metrics_json = {}

    for strategy in simulation.aggregation_strategies:
        detailed_metrics_json[strategy] = {}

        for seed in seeds:
            logger.info(f"============================================================")
            logger.info(f"🔄 Agregação: {strategy} | Seed: {seed}")
            logger.info(f"============================================================")

            client_seed = get_final_seed(target_client_id, seed)
            X_train_target, y_train_target, X_test_target, y_test_target = load_house_client(
                seed=client_seed,
                alpha=settings.client.dirichlet_alpha,
                bins=settings.client.number_of_bins_for_dirichlet,
                percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
            )

            # ------------------------------------------------------------------
            # A. Treino e SHAP dos Modelos BASELINE Não-Privados (epsilon = -1.0)
            # ------------------------------------------------------------------
            logger.info("🌲 Treinando Modelos Baseline Não-Privados (epsilon = -1.0)...")
            base_local_model = fit_local_tree(X_train_target, y_train_target, -1.0, client_seed)
            base_global_model = build_global_model(strategy, baseline_setting, seed, num_clients)

            exp_base_local = shap.TreeExplainer(base_local_model)
            shap_base_local = exp_base_local.shap_values(X_test_target)

            exp_base_global = shap.TreeExplainer(base_global_model)
            shap_base_global = exp_base_global.shap_values(X_test_target)

            # ------------------------------------------------------------------
            # B. Avaliação para cada Configuração de Epsilon
            # ------------------------------------------------------------------
            for setting in epsilon_settings:
                epsilon = setting.epsilon
                eps_str = f"eps_{epsilon}"

                if eps_str not in detailed_metrics_json[strategy]:
                    detailed_metrics_json[strategy][eps_str] = {}

                logger.info(f"👉 Avaliando Epsilon: {epsilon} | Estratégia: {strategy} | Seed: {seed}")

                # 1. Treino dos modelos sob teste
                if epsilon == -1.0:
                    local_model = base_local_model
                    global_model = base_global_model
                    shap_local = shap_base_local
                    shap_global = shap_base_global
                else:
                    local_model = fit_local_tree(X_train_target, y_train_target, epsilon, client_seed)
                    global_model = build_global_model(strategy, setting, seed, num_clients)

                    exp_local = shap.TreeExplainer(local_model)
                    shap_local = exp_local.shap_values(X_test_target)

                    exp_global = shap.TreeExplainer(global_model)
                    shap_global = exp_global.shap_values(X_test_target)

                # 2. Métricas de Complexidade
                comp_local = compute_complexity_metrics(local_model)
                comp_global = compute_complexity_metrics(global_model)

                # 3. Métricas de Esparsidade
                sparse_local = compute_sparsity_metrics(shap_local)
                sparse_global = compute_sparsity_metrics(shap_global)

                # 4. Métricas de Fidelidade vs Baseline (-1.0)
                fid_local = compute_fidelity_metrics(local_model, base_local_model, X_test_target)
                fid_global = compute_fidelity_metrics(global_model, base_global_model, X_test_target)

                # 5. Monotonicidade / Deletion AUC
                del_auc_local = compute_deletion_auc(local_model, shap_local, X_test_target)
                del_auc_global = compute_deletion_auc(global_model, shap_global, X_test_target)

                # 6. Rank Stability vs Baseline
                rank_local = compute_rank_stability(shap_base_local, shap_local)
                rank_global = compute_rank_stability(shap_base_global, shap_global)

                # 7. Attribution Distance vs Baseline
                dist_local = compute_attribution_distance(shap_base_local, shap_local)
                dist_global = compute_attribution_distance(shap_base_global, shap_global)

                # 8. Local Sensitivity
                sens_local = compute_local_sensitivity(local_model, X_test_target, shap_local, SAMPLE_SIZE_LOCAL_SENSITIVITY, NOISE_SCALE_LOCAL_SENSITIVITY)
                sens_global = compute_local_sensitivity(global_model, X_test_target, shap_global, SAMPLE_SIZE_LOCAL_SENSITIVITY, NOISE_SCALE_LOCAL_SENSITIVITY)

                # 9. Federated Explanation Gap (Local vs Global no mesmo cliente)
                exp_gap = compute_explanation_gap(shap_local, shap_global)

                # Estrutura detalhada de resultados da combinação
                eval_record = {
                    "strategy": strategy,
                    "epsilon": epsilon,
                    "seed": seed,
                    "local": {
                        **comp_local,
                        **sparse_local,
                        **fid_local,
                        "deletion_auc": del_auc_local,
                        **rank_local,
                        **dist_local,
                        "local_sensitivity": sens_local,
                    },
                    "global": {
                        **comp_global,
                        **sparse_global,
                        **fid_global,
                        "deletion_auc": del_auc_global,
                        **rank_global,
                        **dist_global,
                        "local_sensitivity": sens_global,
                    },
                    "federated_explanation_gap": exp_gap
                }

                detailed_metrics_json[strategy][eps_str][f"seed_{seed}"] = eval_record

                # Registros sumarizados para DataFrame / CSV
                for model_level in ["local", "global"]:
                    metrics_dict = eval_record[model_level]
                    row = {
                        "strategy": strategy,
                        "epsilon": epsilon,
                        "seed": seed,
                        "model_level": model_level,
                        **metrics_dict,
                        **exp_gap
                    }
                    all_summary_rows.append(row)

    # --------------------------------------------------------------------------
    # C. Exportação dos Resultados
    # --------------------------------------------------------------------------
    logger.info("💾 Exportando relatórios JSON...")
    json_detailed_path = results_base / "explainability_detailed.json"
    with open(json_detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics_json, f, indent=4)

    json_summary_path = results_base / "explainability_summary.json"
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary_rows, f, indent=4)

    logger.info(f"✅ Avaliação concluída com sucesso!")
    logger.info(f"📄 Arquivo JSON Detalhado exportado para: {json_detailed_path}")
    logger.info(f"📊 Arquivo JSON Sumarizado exportado para: {json_summary_path}")


if __name__ == "__main__":
    run_explainability_eval()
