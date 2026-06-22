import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.scripts.settings import graphics

def plot_ensemble_analysis_graphics():
    folder_name = "ensemble_analysis"
    input_file = paths.results_folder / "side_tests" / folder_name / "ensemble_results.json"
    output_dir = paths.graphics_path / folder_name
    
    if not input_file.exists():
        print(f"[!] Arquivo de resultados do Ensemble não encontrado em: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        global_results = json.load(f)

    for scenario_key, pct_data in global_results.items():
        # Desfaz a chave composta criada na simulação
        epsilon, strategy = scenario_key.split("__")
        
        plot_pcts = []
        plot_avg_r2 = []
        plot_avg_rel_mse = []
        
        # Ordena as chaves de porcentagem de forma decrescente (1.0 até 0.1)
        sorted_pcts = sorted([float(k) for k in pct_data.keys()], reverse=True)
        
        for pct in sorted_pcts:
            pct_str = str(pct)
            avg_r2 = np.mean(pct_data[pct_str]['r2'])
            avg_rel_mse = np.mean(pct_data[pct_str]['rel_mse'])
            
            plot_pcts.append(int(pct * 100))
            plot_avg_r2.append(avg_r2)
            plot_avg_rel_mse.append(avg_rel_mse)

        # --- Renderização Gráfica Integrada ao Sistema de Estilos ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Degradação do Ensemble sob Privacidade Diferencial\n(Epsilon: {epsilon} | Estratégia: {strategy})", 
            fontsize=graphics.fontsize, 
            fontweight=graphics.fontweight
        )
        
        # Subplot 1: R² Score
        ax1.plot(plot_pcts, plot_avg_r2, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Média R²')
        ax1.set_title("Evolução da Generalização (R² Score)", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax1.set_xlabel("% de Árvores Preservadas no Modelo Global", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax1.set_ylabel("R² Score (Maior é melhor)", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax1.set_xticks(plot_pcts)
        ax1.set_xlim(105, 5)
        ax1.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
        ax1.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
        
        # Subplot 2: Variação Relativa do MSE
        ax2.plot(plot_pcts, plot_avg_rel_mse, marker='s', linestyle='-', color='#d62728', linewidth=2, label='Var. Relativa MSE')
        ax2.set_title("Aumento Percentual do Erro (Var. Relativa MSE)", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax2.set_xlabel("% de Árvores Preservadas no Modelo Global", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax2.set_ylabel("Aumento do Erro em relação ao baseline de 100% (%)", fontsize=graphics.fontsize - 2, fontweight=graphics.fontweight)
        ax2.set_xticks(plot_pcts)
        ax2.set_xlim(105, 5)
        ax2.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
        ax2.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
        
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"degradacao_ensemble_eps_{epsilon}_{strategy}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()