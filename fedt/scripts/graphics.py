import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def outliers_manager(remove_outliers, aggregated_data):
    if remove_outliers:
        for strategy in aggregated_data:
            for epsilon in aggregated_data[strategy]:
                group_values = aggregated_data[strategy][epsilon]

                if len(group_values) >= 4:
                    q1, q3 = np.percentile(group_values, [25, 75])
                    interquartile_range = q3 - q1

                    lim_inf_mod, lim_sup_mod = q1 - 1.5 * interquartile_range, q3 + 1.5 * interquartile_range
                    lim_inf_ext, lim_sup_ext = q1 - 3.0 * interquartile_range, q3 + 3.0 * interquartile_range

                    clean_data = []
                    for value in group_values:
                        is_extremo = value < lim_inf_ext or value > lim_sup_ext
                        is_moderado = (value < lim_inf_mod or value > lim_sup_mod) and not is_extremo
                        
                        if remove_outliers == 'ambos' and (is_extremo or is_moderado): continue
                        if remove_outliers == 'extremos' and is_extremo: continue
                        if remove_outliers == 'moderados' and is_moderado: continue
                        
                        clean_data.append(value)

                    aggregated_data[strategy][epsilon] = clean_data

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
                    continue

            rounds = [key for key in json_data.keys() if key.startswith("round_")]
            if not rounds:
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
                if target_metric in server_data:
                    aggregated_data[strategy][epsilon].append(server_data[target_metric])

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

def line_plot(aggregated_data, target_strategy, metric_name):
    aggregated_data = load_simulation_data(
        base_path=caminho_base, 
        target_metric=metrica_alvo,
        user_type="clients",
        remove_outliers="extremos"
    )

    data = extract_data_for_plot(aggregated_data, target_strategy, metric_name)

    if not data:
        return

    means, deviations, labels, data_plot = data

    plt.figure(figsize=(9, 6))
    plt.errorbar(labels, means, yerr=deviations, fmt='-o', color='b', capsize=5, capthick=2, elinewidth=2, markersize=8, label=f'Estratégia: {target_strategy}')
    plt.title(f"Impacto da Privacidade na Métrica {metric_name.upper()}", fontsize=14)
    plt.xlabel("Nível de Privacidade (Epsilon)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric_name}_linha.pdf")
    plt.close()

def bar_plot(target_strategy, metric_name, translation_dictionary):
    aggregated_data = load_simulation_data(
        base_path=caminho_base, 
        target_metric=metrica_alvo,
        user_type="clients",
        remove_outliers="extremos"
    )

    data = extract_data_for_plot(aggregated_data, target_strategy, metric_name)

    if not data:
        return

    means, deviations, labels, data_plot = data
    labels = rename_epsilon(labels)

    plt.figure(figsize=(9, 6))
    plt.bar(labels, means, yerr=deviations, color='skyblue', edgecolor='navy', capsize=6, alpha=0.85, label=f'Estratégia: {target_strategy}')
    plt.xlabel("Privacy Level (ε)", fontsize=12)
    plt.ylabel(translation_dictionary[metric_name], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric_name}_barras.pdf")
    plt.close()


def box_plot(target_strategy, metric_name, translation_dictionary, user_type, remove_outliers):
    aggregated_data = load_simulation_data(
        base_path=caminho_base, 
        target_metric=metric_name,
        user_type="clients",
        remove_outliers="extremos"
    )

    data = extract_data_for_plot(aggregated_data, target_strategy, metric_name)

    if not data:
        return

    means, deviations, labels, data_plot = data
    labels = rename_epsilon(labels, translation_dictionary)

    plt.figure(figsize=(9, 6))
    plt.boxplot(data_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2))
    
    plt.xlabel("Privacy Level (ε)", fontsize=fontsize, fontweight=fontweight)
    plt.ylabel(translation_dictionary[metric_name], fontsize=fontsize, fontweight=fontweight)

    plt.tick_params(axis='both', labelsize=ticks_fontsize)

    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{target_strategy}_{metric_name}_boxplot.pdf")
    plt.close()

if __name__ == "__main__":
    translation_dictionary = {
        "initial_mse" : "Local Model MSE (Wh²)",
        "initial_rmse" : "Local Model RMSE (Wh)",
        "final_mse" : "Global Model MSE (Wh²)",
        "final_rmse" : "Global Model RMSE (Wh)",
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

    fontsize = 16
    fontweight = "bold"
    ticks_fontsize = 13

    caminho_base = "results"
    remove_outliers = "extremos" # Opções: None, "moderados", "extremos", "ambos"
        
    box_plot(
        target_strategy="threshold_trees", 
        metric_name="initial_rmse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="threshold_trees", 
        metric_name="final_rmse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="all_trees", 
        metric_name="initial_rmse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="all_trees", 
        metric_name="final_rmse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )

    box_plot(
        target_strategy="threshold_trees", 
        metric_name="initial_mse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="threshold_trees", 
        metric_name="final_mse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="all_trees", 
        metric_name="initial_mse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )
    box_plot(
        target_strategy="all_trees", 
        metric_name="final_mse", 
        translation_dictionary=translation_dictionary,
        user_type="clients",
        remove_outliers=remove_outliers
    )

    print("Gráficos gerados e salvos com sucesso!")