import json
import numpy as np
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.scripts_for_graphics.settings import graphics
from fedt.scripts_for_graphics.utils import remove_outliers_from_list

import logging
logger = logging.getLogger("GRAPHICS")

def sort_epsilons(e):
    if str(e) in ["-1.0", "-1", "no-diff-privacy"]:
        return float('inf')
    return float(e)

def dra_outliers_manager(remove_outliers, data_dict):
    if not remove_outliers:
        return data_dict

    cleaned_dict = {}
    for epsilon, group_values in data_dict.items():
        cleaned_dict[epsilon] = remove_outliers_from_list(group_values, remove_outliers)
    return cleaned_dict

def load_external_dra_data(file_path=None, remove_outliers=None):
    if file_path is None:
        file_path = paths.base_path / "external_results" / "dra_results.json"
        
    if not file_path.exists():
        logger.warning(f"Arquivo de resultados externos DRA não encontrado em: {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    if remove_outliers:
        for metric in results_data:
            results_data[metric] = dra_outliers_manager(remove_outliers, results_data[metric])

    return results_data

def extract_data_for_lineplot(data_dict):
    sorted_eps = sorted(data_dict.keys(), key=sort_epsilons, reverse=True)
    means = []
    stds = []
    labels = []

    for eps in sorted_eps:
        values = data_dict[eps]
        means.append(np.mean(values))
        stds.append(np.std(values))
        if str(eps) in ["-1.0", "-1", "no-diff-privacy"]:
            labels.append(graphics.labels.epsilon.no_diff_privacy)
        else:
            labels.append(str(eps))

    return sorted_eps, means, stds, labels

def dra_line_plot_with_external(fedt_dict, sbdt_dict, file_name, y_label):
    if not fedt_dict:
        logger.warning(f"Dados FEDT insuficientes para plotar linha DRA: {file_name}")
        return

    fedt_eps, fedt_means, fedt_stds, fedt_labels = extract_data_for_lineplot(fedt_dict)
    
    labels = fedt_labels
    x_positions = {eps: i for i, eps in enumerate(fedt_eps)}
    x_all = np.arange(len(labels))

    plt.figure(figsize=tuple(graphics.normal_figsize))

    plt.errorbar(
        x_all, fedt_means, yerr=fedt_stds,
        marker=graphics.client.marker,
        linestyle=graphics.client.linestyle,
        color=graphics.client.color,
        linewidth=graphics.lines.linewidth,
        capsize=graphics.lines.capsize,
        label="FEDT"
    )

    if sbdt_dict:
        sbdt_eps, sbdt_means_raw, sbdt_stds_raw, _ = extract_data_for_lineplot(sbdt_dict)
        sbdt_x = [x_positions[eps] for eps in sbdt_eps if eps in x_positions]
        sbdt_means = [sbdt_means_raw[i] for i, eps in enumerate(sbdt_eps) if eps in x_positions]
        sbdt_stds = [sbdt_stds_raw[i] for i, eps in enumerate(sbdt_eps) if eps in x_positions]

        plt.errorbar(
            sbdt_x, sbdt_means, yerr=sbdt_stds,
            marker=graphics.sbdt.marker,
            linestyle=graphics.sbdt.linestyle,
            color=graphics.sbdt.color,
            linewidth=graphics.lines.linewidth,
            capsize=graphics.lines.capsize,
            label=graphics.sbdt.label
        )

    plt.xticks(x_all, labels)
    plt.xlabel(graphics.labels.x.privacy_level, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
    plt.ylabel(y_label, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)

    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    plt.legend(fontsize=graphics.legend_fontsize)
    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()

def dra_boxplot(result_dict, file_name, y_label):
    data_plot = []
    labels = []

    sorted_eps = sorted(result_dict.keys(), key=sort_epsilons, reverse=True)

    for epsilon in sorted_eps:
        values = result_dict[epsilon]
        if len(values) > 0:
            data_plot.append(values)
            if str(epsilon) in ["-1.0", "-1", "no-diff-privacy"]:
                labels.append(graphics.labels.epsilon.no_diff_privacy)
            else:
                labels.append(str(epsilon))

    if not data_plot:
        logger.critical(f"[!] Não há dados válidos para plotar o gráfico: {file_name}")
        return

    plt.figure(figsize=tuple(graphics.normal_figsize))
    plt.boxplot(
        data_plot, 
        tick_labels=labels, 
        patch_artist=True, 
        boxprops=dict(facecolor=graphics.boxplot.box_facecolor, color=graphics.boxplot.box_color), 
        medianprops=dict(color=graphics.boxplot.median_color, linewidth=graphics.boxplot.median_linewidth)
    )
    
    plt.xlabel(graphics.labels.x.privacy_level, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
    plt.ylabel(y_label, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha, axis='y')
    plt.tight_layout()
    
    plt.savefig(file_name)
    plt.close()

def plot_data_reconstruction_attack_graphics():
    folder_name = "data_reconstruction_attack"
    input_dir = paths.results_folder / "side_tests" / folder_name
        
    file_path = input_dir / "dra_results.json"
    
    if not file_path.exists():
        logger.critical(f"[!] Arquivo de resultados DRA não encontrado em: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    result_dict_mse = results_data.get("mse", {})
    result_dict_rmse = results_data.get("rmse", {})

    opcao_filtragem = graphics.remove_outliers
    result_dict_mse = dra_outliers_manager(opcao_filtragem, result_dict_mse)
    result_dict_rmse = dra_outliers_manager(opcao_filtragem, result_dict_rmse)

    external_data = load_external_dra_data(remove_outliers=opcao_filtragem)
    sbdt_mse = external_data.get("mse", {})
    sbdt_rmse = external_data.get("rmse", {})

    output_dir = paths.graphics_path / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dra_boxplot(
        result_dict_mse,
        file_name=f"{output_dir}/recovery_attack_MSE_Y.pdf",
        y_label=graphics.labels.y.dra_mse
    )
    dra_boxplot(
        result_dict_rmse,
        file_name=f"{output_dir}/recovery_attack_RMSE_Y.pdf",
        y_label=graphics.labels.y.dra_rmse
    )

    dra_line_plot_with_external(
        result_dict_mse,
        sbdt_mse,
        file_name=f"{output_dir}/recovery_attack_MSE_Y_lineplot.pdf",
        y_label=graphics.labels.y.dra_mse
    )
    dra_line_plot_with_external(
        result_dict_rmse,
        sbdt_rmse,
        file_name=f"{output_dir}/recovery_attack_RMSE_Y_lineplot.pdf",
        y_label=graphics.labels.y.dra_rmse
    )