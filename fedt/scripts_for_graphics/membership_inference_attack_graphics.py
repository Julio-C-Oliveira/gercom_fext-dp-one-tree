import json
import numpy as np
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.scripts.settings import graphics
from fedt.scripts_for_graphics.utils import remove_outliers_from_list

import logging
logger = logging.getLogger("GRAPHICS")

def mia_outliers_manager(remove_outliers, data_dict):
    if not remove_outliers:
        return data_dict

    cleaned_dict = {}
    for epsilon, group_values in data_dict.items():
        cleaned_dict[epsilon] = remove_outliers_from_list(group_values, remove_outliers)
    return cleaned_dict

def mia_boxplot(result_dict, file_name, y_label):
    data_plot = []
    labels = []

    for epsilon, values in result_dict.items():
        if len(values) > 0:
            data_plot.append(values)
            if str(epsilon) == "-1.0" or epsilon == -1.0:
                labels.append("No Diff Priv")
            else:
                labels.append(str(epsilon))

    if not data_plot:
        logger.critical(f"[!] Não há dados válidos para plotar o gráfico: {file_name}")
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
    
    plt.savefig(file_name)
    plt.close()

def plot_membership_inference_attack_graphics():
    folder_name = "membership_inference_attack"
    input_dir = paths.results_folder / "side_tests" / folder_name
        
    file_path = input_dir / "mia_results.json"
    
    if not file_path.exists():
        logger.critical(f"[!] Arquivo de resultados MIA não encontrado em: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    result_dict_accuracy = results_data.get("accuracy", {})
    result_dict_auc = results_data.get("auc", {})

    opcao_filtragem = graphics.remove_outliers
    result_dict_accuracy = mia_outliers_manager(opcao_filtragem, result_dict_accuracy)
    result_dict_auc = mia_outliers_manager(opcao_filtragem, result_dict_auc)

    output_dir = paths.graphics_path / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mia_boxplot(
        result_dict_accuracy,
        file_name=f"{output_dir}/mia_attack_Accuracy.pdf",
        y_label="MIA Attack Accuracy"
    )
    mia_boxplot(
        result_dict_auc,
        file_name=f"{output_dir}/mia_attack_AUC_ROC.pdf",
        y_label="MIA Attack AUC-ROC"
    )