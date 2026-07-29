import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from fedt.app.settings import paths
from fedt.scripts.settings import graphics

logger = logging.getLogger("GRAPHICS")

# Paleta de cores distintas para o Modelo Local e as 4 Estratégias Globais
SERIES_COLORS = {
    "local": "#9467bd",                       # Roxo para o modelo local
    "ensemble_all_trees": "#1f77b4",        # Azul
    "ensemble_threshold_trees": "#ff7f0e",  # Laranja
    "merge_all_trees": "#2ca02c",           # Verde
    "merge_threshold_trees": "#d62728"      # Vermelho
}

SERIES_MARKERS = {
    "local": "v",
    "ensemble_all_trees": "o",
    "ensemble_threshold_trees": "s",
    "merge_all_trees": "^",
    "merge_threshold_trees": "D"
}

SERIES_LABELS = {
    "local": "Local Model (Client)",
    "ensemble_all_trees": "Ensemble All Trees",
    "ensemble_threshold_trees": "Ensemble Threshold Trees",
    "merge_all_trees": "Merge All Trees",
    "merge_threshold_trees": "Merge Threshold Trees"
}

METRIC_CONFIGS = {
    "hoyer_sparsity": {
        "title": "Hoyer Sparsity Index",
        "ylabel": "Hoyer Sparsity Index",
        "filename": "hoyer_sparsity",
        "stat": "median"
    },
    "gini_index": {
        "title": "Gini Index of Attribution",
        "ylabel": "Gini Index",
        "filename": "gini_index",
        "stat": "median"
    },
    "mae_fidelity": {
        "title": "Prediction Fidelity (MAE vs. Non-Private Baseline)",
        "ylabel": "MAE",
        "filename": "fidelity_mae",
        "stat": "median"
    },
    "spearman_rank_corr": {
        "title": "Feature Rank Stability (Spearman ρ)",
        "ylabel": "Spearman Correlation (ρ)",
        "filename": "rank_spearman",
        "stat": "median"
    },
    "jaccard_top3": {
        "title": "Top-3 Feature Rank Stability (Jaccard)",
        "ylabel": "Jaccard Similarity",
        "filename": "rank_jaccard_top3",
        "stat": "mean_std"  # Média e Desvio Padrão conforme solicitado
    },
    "jaccard_top5": {
        "title": "Top-5 Feature Rank Stability (Jaccard)",
        "ylabel": "Jaccard Similarity",
        "filename": "rank_jaccard_top5",
        "stat": "mean_std"  # Média e Desvio Padrão conforme solicitado
    },
    "cosine_distance": {
        "title": "SHAP Attribution Cosine Distance vs. Baseline",
        "ylabel": "Cosine Distance",
        "filename": "attribution_cosine_distance",
        "stat": "median"
    },
    "local_sensitivity": {
        "title": "Local Explanation Sensitivity (Lipschitz)",
        "ylabel": "Sensitivity Ratio",
        "filename": "local_sensitivity",
        "stat": "median"
    },
    "gap_cosine_distance": {
        "title": "Federated Explanation Gap (Cosine Distance)",
        "ylabel": "Cosine Distance (Local vs. Global)",
        "filename": "federated_gap_cosine_distance",
        "stat": "median"
    },
    "gap_jaccard_top3": {
        "title": "Federated Explanation Gap (Top-3 Jaccard)",
        "ylabel": "Top-3 Jaccard Similarity (Local vs. Global)",
        "filename": "federated_gap_jaccard_top3",
        "stat": "mean_std"
    },
    "gap_spearman_rank_corr": {
        "title": "Federated Explanation Gap (Spearman ρ)",
        "ylabel": "Spearman Correlation (Local vs. Global)",
        "filename": "federated_gap_spearman_rank_corr",
        "stat": "median"
    }
}


def sort_epsilons(eps_set):
    """
    Ordena os epsilons corretamente no eixo X:
    -1.0 (No Diff Privacy) primeiro, seguido pelos epsilons positivos em ordem DECRESCENTE
    (ex: -1.0, 10.0, 7.0, 5.0, 3.0, 1.0, 0.75, 0.5, 0.25, 0.1).
    """
    positive_eps = sorted([e for e in eps_set if e != -1.0], reverse=True)
    if -1.0 in eps_set:
        return [-1.0] + positive_eps
    return positive_eps


def load_summary_data(json_file):
    """Carrega o JSON sumarizado e organiza os epsilons na ordem correta."""
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    eps_set = set(rec["epsilon"] for rec in records)
    eps_list = sort_epsilons(eps_set)

    # Rótulo para o eixo X
    labels = ["No Diff Priv." if e == -1.0 else str(e) for e in eps_list]

    return records, eps_list, labels


def compute_series_stats(records, target_metric, model_level, strategy, eps_list, stat_type="median"):
    """
    Calcula as estatísticas para a série:
    - stat_type="median": Mediana com barras de erro assimétricas Q1 (25%) e Q3 (75%).
    - stat_type="mean_std": Média com desvio padrão (STD).
    """
    center_vals = []
    yerr_list = []

    for eps in eps_list:
        if model_level == "local":
            matching_recs = [
                rec[target_metric]
                for rec in records
                if rec["model_level"] == "local"
                and rec["epsilon"] == eps
                and target_metric in rec
                and not np.isnan(rec[target_metric])
            ]
        else:
            matching_recs = [
                rec[target_metric]
                for rec in records
                if rec["model_level"] == model_level
                and rec["strategy"] == strategy
                and rec["epsilon"] == eps
                and target_metric in rec
                and not np.isnan(rec[target_metric])
            ]

        if len(matching_recs) > 0:
            if stat_type == "mean_std":
                mean_val = float(np.mean(matching_recs))
                std_val = float(np.std(matching_recs))
                center_vals.append(mean_val)
                yerr_list.append(std_val)
            else: # median
                med = float(np.median(matching_recs))
                q1 = float(np.percentile(matching_recs, 25))
                q3 = float(np.percentile(matching_recs, 75))
                center_vals.append(med)
                yerr_list.append([med - q1, q3 - med])
        else:
            center_vals.append(np.nan)
            if stat_type == "mean_std":
                yerr_list.append(np.nan)
            else:
                yerr_list.append([np.nan, np.nan])

    center_arr = np.array(center_vals)
    if stat_type == "mean_std":
        yerr_arr = np.array(yerr_list)
    else:
        # Formato (2, N) para errorbar assimétrico
        lower = [y[0] for y in yerr_list]
        upper = [y[1] for y in yerr_list]
        yerr_arr = [np.array(lower), np.array(upper)]

    return center_arr, yerr_arr


def render_unified_line_plot(x_indices, series_dict, x_labels, title, ylabel, output_path):
    """Renderiza gráfico de linhas unificado com Mediana/Q1-Q3 ou Média/STD."""
    fig, ax = plt.subplots(figsize=tuple(graphics.normal_figsize))

    for series_key, data in series_dict.items():
        color = SERIES_COLORS.get(series_key, "#333333")
        marker = SERIES_MARKERS.get(series_key, "o")
        label = SERIES_LABELS.get(series_key, series_key)

        ax.errorbar(
            x_indices,
            data["center"],
            yerr=data["yerr"],
            marker=marker,
            linestyle="-",
            color=color,
            linewidth=2,
            capsize=4,
            capthick=1.2,
            label=label
        )

    ax.set_xlabel("Privacy Level (ε)", fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    ax.set_ylabel(ylabel, fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=graphics.ticks_fontsize)
    ax.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    ax.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
    ax.legend(fontsize=graphics.ticks_fontsize - 1, loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Gráfico unificado salvo: {output_path.name}")


def plot_explainability_eval_graphics():
    input_file = paths.results_folder / "side_tests" / "explainability_eval" / "explainability_summary.json"
    output_dir = paths.graphics_path / "explainability_eval"

    if not input_file.exists():
        logger.critical(f"[!] Arquivo de resultados não encontrado em: {input_file}")
        return

    records, eps_list, x_labels = load_summary_data(input_file)
    x_indices = np.arange(len(eps_list))

    global_strategies = [
        "ensemble_all_trees",
        "ensemble_threshold_trees",
        "merge_all_trees",
        "merge_threshold_trees"
    ]

    # Lista de métricas unificadas (Local + 4 Globais na mesma figura)
    unified_metrics = [
        "hoyer_sparsity",
        "gini_index",
        "mae_fidelity",
        "spearman_rank_corr",
        "jaccard_top3",
        "jaccard_top5",
        "cosine_distance",
        "local_sensitivity"
    ]

    # --------------------------------------------------------------------------
    # 1. GRÁFICOS UNIFICADOS (LOCAL MODEL + 4 GLOBAL STRATEGIES)
    # --------------------------------------------------------------------------
    for metric_key in unified_metrics:
        cfg = METRIC_CONFIGS[metric_key]
        series_dict = {}

        # Serie do Modelo Local
        center_loc, yerr_loc = compute_series_stats(
            records, metric_key, "local", "ensemble_all_trees", eps_list, stat_type=cfg["stat"]
        )
        series_dict["local"] = {"center": center_loc, "yerr": yerr_loc}

        # Series das 4 Estratégias Globais
        for strat in global_strategies:
            center_glob, yerr_glob = compute_series_stats(
                records, metric_key, "global", strat, eps_list, stat_type=cfg["stat"]
            )
            series_dict[strat] = {"center": center_glob, "yerr": yerr_glob}

        out_path = output_dir / f"{cfg['filename']}_vs_epsilon.pdf"
        render_unified_line_plot(x_indices, series_dict, x_labels, cfg["title"], cfg["ylabel"], out_path)

    # --------------------------------------------------------------------------
    # 2. GRÁFICOS DO GAP DE EXPLICABILIDADE FEDERADA (AS 4 ESTRATÉGIAS GLOBAIS)
    # --------------------------------------------------------------------------
    gap_metrics = ["gap_cosine_distance", "gap_jaccard_top3", "gap_spearman_rank_corr"]

    for metric_key in gap_metrics:
        cfg = METRIC_CONFIGS[metric_key]
        series_dict = {}

        for strat in global_strategies:
            center_gap, yerr_gap = compute_series_stats(
                records, metric_key, "global", strat, eps_list, stat_type=cfg["stat"]
            )
            series_dict[strat] = {"center": center_gap, "yerr": yerr_gap}

        out_path = output_dir / f"{cfg['filename']}_vs_epsilon.pdf"
        render_unified_line_plot(x_indices, series_dict, x_labels, cfg["title"], cfg["ylabel"], out_path)

    logger.info(f"✅ Todos os gráficos foram unificados e salvos com sucesso em: {output_dir}")


if __name__ == "__main__":
    plot_explainability_eval_graphics()