import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.simulation.settings import simulation

import logging
logger = logging.getLogger("GRAPHICS")

def generate_visual_plots(data_path, output_dir, prefix):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Reconstrução dos dados
    shap_values = np.array(data["shap_values"])
    base_value = data["expected_value"]
    feature_names = data["feature_names"]
    
    # É fundamental transformar X_test de volta num DataFrame para os dependence_plots funcionarem via nome de coluna
    X_test = pd.DataFrame(data["X_test"], columns=feature_names)

    # print(f"  [{prefix}] Gerando Gráficos SHAP...")

    # --- 1. Summary Plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_dir / f"{prefix}_summary_global.pdf", bbox_inches='tight')
    plt.close('all')

    # --- 2. Force Plot (Para a primeira amostra) ---
    shap.force_plot(base_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.savefig(output_dir / f"{prefix}_force_plot.pdf", bbox_inches='tight')
    plt.close('all')

    # --- 3. Dependence Plots ---
    for col in ["T_out", "RH_out"]:
        if col in X_test.columns:
            plt.figure()
            shap.dependence_plot(col, shap_values, X_test, show=False)
            plt.savefig(output_dir / f"{prefix}_dependence_{col.lower()}.pdf", bbox_inches='tight')
            plt.close('all')

    # --- 4. Decision Plot ---
    plt.figure()
    shap.decision_plot(base_value, shap_values, X_test, show=False, ignore_warnings=True)
    plt.savefig(output_dir / f"{prefix}_decision_plot.pdf", bbox_inches='tight')
    plt.close('all')

    # --- 5. Waterfall Plot ---
    exp = shap.Explanation(
        values=shap_values[0], 
        base_values=base_value, 
        data=X_test.iloc[0], 
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.savefig(output_dir / f"{prefix}_waterfall_single.pdf", bbox_inches='tight')
    plt.close('all')

    # --- 6. Heatmap ---
    exp_all = shap.Explanation(
        values=shap_values, 
        base_values=base_value, 
        data=X_test, 
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.heatmap(exp_all[:100], show=False)
    plt.savefig(output_dir / f"{prefix}_heatmap.pdf", bbox_inches='tight')
    plt.close('all')

def plot_shap_analysis_graphics():
    input_base = paths.results_folder / "side_tests" / "shap_analysis"
    output_base = paths.graphics_path / "shap_analysis"

    for strategy in simulation.aggregation_strategies:
        for setting in simulation.epsilon_settings:
            logger.debug(f"Gerando gráficos SHAP para a estratégia {strategy}, eps {setting.epsilon}")
            for seed in [simulation.seeds[8]]:
                folder_rel = f"{strategy}/eps_{setting.epsilon}/seed_{seed}"
                in_dir = input_base / folder_rel
                out_dir = output_base / folder_rel
                
                if not in_dir.exists():
                    logger.debug(f"Diretório não encontrado, ignorando: {in_dir}")
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                local_json_path = in_dir / "shap_local.json"
                global_json_path = in_dir / "shap_global.json"

                if local_json_path.exists():
                    generate_visual_plots(local_json_path, out_dir, "LOCAL")
                if global_json_path.exists():
                    generate_visual_plots(global_json_path, out_dir, "GLOBAL")