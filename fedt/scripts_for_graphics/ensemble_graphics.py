import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.scripts.settings import graphics

import logging
logger = logging.getLogger("GRAPHICS")

def plot_ensemble_analysis_graphics():
    folder_name = "ensemble_analysis"
    input_file = paths.results_folder / "side_tests" / folder_name / "ensemble_results.json"
    output_dir = paths.graphics_path / folder_name
    
    if not input_file.exists():
        logger.critical(f"[!] Arquivo de resultados do Ensemble não encontrado em: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        global_results = json.load(f)

    for scenario_key, pct_data in global_results.items():
        # Desfaz a chave composta criada na simulação
        epsilon, strategy = scenario_key.split("__")

        logger.debug(f"Processando cenário: Epsilon = {epsilon}, Estratégia = {strategy}")
        
        plot_pcts = []
        plot_avg_rel_mse = []
        
        # Ordena as chaves de porcentagem de forma decrescente (1.0 até 0.1)
        sorted_pcts = sorted([float(k) for k in pct_data.keys()], reverse=True)
        
        for pct in sorted_pcts:
            pct_str = str(pct)
            avg_rel_mse = np.mean(pct_data[pct_str]['rel_mse'])
            
            plot_pcts.append(int(pct * 100))
            plot_avg_rel_mse.append(avg_rel_mse)

        # --- Renderização Gráfica Integrada ao Sistema de Estilos ---
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot: Variação Relativa do MSE
        ax.plot(plot_pcts, plot_avg_rel_mse, marker='s', linestyle='-', color='#d62728', linewidth=2, label='Relative MSE Var.')
        ax.set_xlabel("% of Trees Preserved in the Global Model", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax.set_ylabel("Error Increase Relative to Baseline %", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax.set_xticks(plot_pcts)
        ax.set_xlim(105, 5)
        ax.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
        ax.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
        
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"degradacao_ensemble_eps_{epsilon}_{strategy}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráfico salvo: {filename.name}")