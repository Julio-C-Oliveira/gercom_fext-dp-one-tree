import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from fedt.app.settings import paths
from fedt.scripts.settings import graphics
from fedt.scripts_for_graphics.utils import remove_outliers_from_list

from fedt.simulation.settings import simulation

import logging
logger = logging.getLogger("GRAPHICS")

def outliers_manager(remove_outliers, aggregated_data):
    if remove_outliers:
        for strategy in aggregated_data:
            for epsilon in aggregated_data[strategy]:
                group_values = aggregated_data[strategy][epsilon]
                aggregated_data[strategy][epsilon] = remove_outliers_from_list(group_values, remove_outliers)
    return aggregated_data

def load_simulation_data(base_path, target_metric, user_type, remove_outliers):
    aggregated_data = defaultdict(lambda: defaultdict(list))
    strategies = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
    
    for strategy in strategies:
        server_path = os.path.join(base_path, strategy, "server")
        if not os.path.exists(server_path):
            continue

        json_files = [file for file in os.listdir(server_path) if file.endswith(".json")]

        for file in json_files:
            name = file.replace(".json", "")
            prefix_len = len(strategy) + 1
            final_name = name[prefix_len:]

            parts = final_name.rsplit("_", 1)
            if len(parts) != 2:
                continue

            epsilon = parts[0]
            file_path = os.path.join(server_path, file)

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                except json.JSONDecodeError:
                    logger.critical(f"Erro ao decodificar JSON no arquivo {file_path}: {e}")
                    continue

            rounds = [key for key in json_data.keys() if key.startswith("round_")]
            if not rounds:
                logger.warning(f"Nenhum 'round_' encontrado no arquivo {file_path}, ignorando.")
                continue

            rounds.sort(key=lambda x: int(x.split("_")[1]))
            last_round = rounds[-1]

            if user_type == "clients":
                clients_data = json_data[last_round].get("clients", {})
                values = [c_data[target_metric] for c_data in clients_data.values() if target_metric in c_data]
                if values:
                    aggregated_data[strategy][epsilon].append(np.mean(values))

            elif user_type == "server":
                server_data = json_data[last_round].get("server", {})
                server_metric_key = target_metric
                if target_metric not in server_data and target_metric.startswith("cross_validation_"):
                    suffix = target_metric.replace("cross_validation_", "")
                    alt_key = f"global_model_cv_{suffix}"
                    if alt_key in server_data:
                        server_metric_key = alt_key

                if server_metric_key in server_data:
                    aggregated_data[strategy][epsilon].append(server_data[server_metric_key])

    return outliers_manager(remove_outliers, aggregated_data)

def sort_epsilons(e):
    if e == "no-diff-privacy": return float('inf')
    return float(e)

def rename_epsilon(epsilons, translation_dictionary):
    return [translation_dictionary[i] for i in epsilons]

def extract_data_for_plot(aggregated_data, target_strategy, metric_name):
    if target_strategy not in aggregated_data:
        print(f"Estratégia '{target_strategy}' não encontrada para gráfico de linha.")
        return

    strategy_data = aggregated_data[target_strategy]

    sorted_epsilons = sorted(strategy_data.keys(), key=sort_epsilons, reverse=True)
    means, deviations, labels, data_plot = [], [], [], []

    for epsilon in sorted_epsilons:
        values = strategy_data[epsilon]

        means.append(np.mean(values))
        deviations.append(np.std(values))
        data_plot.append(values)

        labels.append(str(epsilon))

    return means, deviations, labels, data_plot

def box_plot(target_strategy, metric_name, translation_dictionary, user_type, remove_outliers):
    caminho_base = paths.results_folder
    aggregated_data = load_simulation_data(
        base_path=caminho_base,
        target_metric=metric_name,
        user_type=user_type,
        remove_outliers=remove_outliers
    )

    data = extract_data_for_plot(aggregated_data, target_strategy, metric_name)

    if not data:
        return

    means, deviations, labels, data_plot = data
    labels = rename_epsilon(labels, translation_dictionary)

    plt.figure(figsize=tuple(graphics.normal_figsize))
    plt.boxplot(data_plot, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2))
    
    plt.xlabel("Privacy Level (ε)", fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    plt.ylabel(translation_dictionary[metric_name], fontsize=graphics.fontsize, fontweight=graphics.fontweight)

    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)

    output_dir = paths.graphics_path / "simulation" 
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{target_strategy}_{metric_name}_boxplot.pdf")
    plt.close()

def line_plot(target_strategy, metric_name, translation_dictionary, remove_outliers):
    caminho_base = paths.results_folder
    clients_aggregated = load_simulation_data(
        base_path=caminho_base,
        target_metric=metric_name,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    server_aggregated = load_simulation_data(
        base_path=caminho_base,
        target_metric=metric_name,
        user_type="server",
        remove_outliers=remove_outliers
    )

    clients_data = extract_data_for_plot(clients_aggregated, target_strategy, metric_name)
    server_data = extract_data_for_plot(server_aggregated, target_strategy, metric_name)

    if not clients_data or not server_data:
        logger.warning(f"Dados insuficientes para a estratégia '{target_strategy}' na métrica '{metric_name}'.")
        return

    c_means, c_deviations, c_labels, _ = clients_data
    s_means, s_deviations, s_labels, _ = server_data

    labels = rename_epsilon(c_labels, translation_dictionary)

    plt.figure(figsize=tuple(graphics.normal_figsize))
    x = np.arange(len(labels))

    plt.errorbar(
        x, c_means, yerr=c_deviations,
        marker='o', linestyle='-', color='blue', linewidth=2, capsize=4,
        label='Clientes'
    )
    plt.errorbar(
        x, s_means, yerr=s_deviations,
        marker='s', linestyle='--', color='red', linewidth=2, capsize=4,
        label='Servidor'
    )

    plt.xticks(x, labels)
    plt.xlabel("Privacy Level (ε)", fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    
    ylabel_text = translation_dictionary.get(metric_name, metric_name)
    plt.ylabel(ylabel_text, fontsize=graphics.fontsize, fontweight=graphics.fontweight)

    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    plt.legend(fontsize=graphics.fontsize - 2)

    output_dir = paths.graphics_path / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{target_strategy}_{metric_name}_lineplot.pdf")
    plt.close()

def combined_line_plot(metric_name, translation_dictionary, remove_outliers):
    caminho_base = paths.results_folder
    
    clients_aggregated = load_simulation_data(
        base_path=caminho_base,
        target_metric=metric_name,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    server_aggregated = load_simulation_data(
        base_path=caminho_base,
        target_metric=metric_name,
        user_type="server",
        remove_outliers=remove_outliers
    )

    if not clients_aggregated or not server_aggregated:
        logger.warning(f"Dados insuficientes para o gráfico combinado da métrica '{metric_name}'.")
        return

    # Extrai o desempenho dos clientes a partir de qualquer estratégia disponível (desempenho local é o mesmo)
    first_strategy = list(clients_aggregated.keys())[0]
    clients_data = extract_data_for_plot(clients_aggregated, first_strategy, metric_name)
    
    if not clients_data:
        logger.warning(f"Não foi possível extrair dados dos clientes para a métrica '{metric_name}'.")
        return

    c_means, c_deviations, c_labels, _ = clients_data
    labels = rename_epsilon(c_labels, translation_dictionary)

    plt.figure(figsize=tuple(graphics.normal_figsize))
    x = np.arange(len(labels))

    # Plotar o cliente uma única vez
    plt.errorbar(
        x, c_means, yerr=c_deviations,
        marker='o', linestyle='-', color='black', linewidth=2, capsize=4,
        label='Clientes'
    )

    # Estilos para diferenciar as estratégias do servidor
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['s', '^', 'v', 'D', 'p', '*', 'X']

    color_idx = 0
    for strategy in simulation.aggregation_strategies:
        if strategy not in server_aggregated:
            continue

        s_data = extract_data_for_plot(server_aggregated, strategy, metric_name)
        if not s_data:
            continue

        s_means, s_deviations, _, _ = s_data
        strategy_label = translation_dictionary.get(strategy, strategy.replace("_", " ").title())
        current_color = colors[color_idx % len(colors)]
        current_marker = markers[color_idx % len(markers)]

        plt.errorbar(
            x, s_means, yerr=s_deviations,
            marker=current_marker, linestyle='--', color=current_color, linewidth=2, capsize=4,
            label=f'Servidor ({strategy_label})'
        )
        color_idx += 1

    plt.xticks(x, labels)
    plt.xlabel("Privacy Level (ε)", fontsize=graphics.fontsize, fontweight=graphics.fontweight)
    
    ylabel_text = translation_dictionary.get(metric_name, metric_name)
    plt.ylabel(ylabel_text, fontsize=graphics.fontsize, fontweight=graphics.fontweight)

    plt.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
    plt.legend(fontsize=graphics.fontsize - 2)

    output_dir = paths.graphics_path / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_strategies_{metric_name}_combined_lineplot.pdf")
    plt.close()

def plot_simulation_graphics():
    translation_dictionary = {
        "initial_mse" : "Local Model MSE (Wh²)",
        "initial_rmse" : "Local Model RMSE (Wh)",
        "final_mse" : "Global Model MSE (Wh²)",
        "final_rmse" : "Global Model RMSE (Wh)",
        "cross_validation_mse" : "Cross Validation MSE (Wh²)",
        "cross_validation_rmse" : "Cross Validation RMSE (Wh)",
        "ensemble_all_trees" : "Ensemble All Trees",
        "ensemble_threshold_trees" : "Ensemble Threshold Trees",
        "merge_all_trees" : "Merge All Trees",
        "merge_threshold_trees" : "Merge Threshold Trees",
        "no-diff-privacy" : "No Diff Priv",
        "10.0" : "10.0",
        "7.0" : "7.0",
        "5.0" : "5.0",
        "3.0" : "3.0",
        "1.0" : "1.0",
        "0.75" : "0.75",
        "0.5" : "0.5",
        "0.25" : "0.25",
        "0.1" : "0.1",
    }

    remove_outliers = graphics.remove_outliers
    
    for strategy in simulation.aggregation_strategies:
        for metric in ["initial_rmse", "final_rmse", "initial_mse", "final_mse"]:
            box_plot(
                target_strategy=strategy, 
                metric_name=metric, 
                translation_dictionary=translation_dictionary,
                user_type="clients",
                remove_outliers=remove_outliers
            )
        for metric in ["cross_validation_rmse", "cross_validation_mse"]:
            line_plot(
                target_strategy=strategy, 
                metric_name=metric, 
                translation_dictionary=translation_dictionary,
                remove_outliers=remove_outliers
            )

    for metric in ["cross_validation_rmse", "cross_validation_mse"]:
        combined_line_plot(
            metric_name=metric,
            translation_dictionary=translation_dictionary,
            remove_outliers=remove_outliers
        )