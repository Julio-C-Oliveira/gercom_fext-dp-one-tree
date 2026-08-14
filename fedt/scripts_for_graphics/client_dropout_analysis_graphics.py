import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fedt.app.settings import paths
from fedt.scripts_for_graphics.settings import graphics

import logging
logger = logging.getLogger("GRAPHICS")

def plot_client_dropout_analysis_graphics():
    folder_name = "client_dropout_analysis"
    input_file = paths.results_folder / "side_tests" / folder_name / "client_dropout_analysis.json"
    output_dir = paths.graphics_path / folder_name
    
    if not input_file.exists():
        logger.critical(f"[!] Arquivo de resultados do Ensemble não encontrado em: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        global_results = json.load(f)

    # 1. Pré-processamento: Agrupar as estratégias pelo valor do Epsilon
    data_by_epsilon = {}
    for scenario_key, pct_data in global_results.items():
        epsilon, strategy = scenario_key.split("____")
        if epsilon not in data_by_epsilon:
            data_by_epsilon[epsilon] = {}
        data_by_epsilon[epsilon][strategy] = pct_data

    for epsilon, strategies in data_by_epsilon.items():
        logger.info(f"Gerando gráficos para Epsilon = {epsilon}")
        
        # Cria a figura para o gráfico combinado de todas as estratégias
        fig_combined, ax_combined = plt.subplots(figsize=tuple(graphics.wide_figsize))
        
        for strategy, pct_data in strategies.items():
            logger.debug(f"Processando estratégia individual: {strategy}")
            
            plot_pcts = []
            plot_median_rel_mse = []
            plot_q1_rel_mse = []
            plot_q3_rel_mse = []
            
            # Ordena as chaves de porcentagem de forma decrescente (1.0 até 0.1)
            sorted_pcts = sorted([float(k) for k in pct_data.keys()], reverse=True)
            
            for pct in sorted_pcts:
                pct_str = str(pct)
                data_array = pct_data[pct_str]['rel_mse']
                
                # Calcula a mediana e os quartis (Q1 = 25%, Q3 = 75%)
                median_val = np.median(data_array)
                q1_val = np.percentile(data_array, 25)
                q3_val = np.percentile(data_array, 75)
                
                plot_pcts.append(int(pct * 100))
                plot_median_rel_mse.append(median_val)
                plot_q1_rel_mse.append(q1_val)
                plot_q3_rel_mse.append(q3_val)

            # Converte para arrays numpy
            plot_pcts = np.array(plot_pcts)
            plot_median_rel_mse = np.array(plot_median_rel_mse)
            plot_q1_rel_mse = np.array(plot_q1_rel_mse)
            plot_q3_rel_mse = np.array(plot_q3_rel_mse)

            # Distância relativa do ponto central para as barras de erro
            lower_error = plot_median_rel_mse - plot_q1_rel_mse
            upper_error = plot_q3_rel_mse - plot_median_rel_mse
            yerr_asymmetric = [lower_error, upper_error]

            # Obtém a cor da estratégia diretamente da configuração
            strategy_cfg = graphics.strategies.get(strategy)
            s_color = strategy_cfg.color if strategy_cfg else '#333333'
            s_label = strategy_cfg.label if strategy_cfg else strategy.replace('_', ' ').title()

            # --- 2. Renderização do Gráfico Individual ---
            fig_ind, ax_ind = plt.subplots(figsize=tuple(graphics.normal_figsize))
            ax_ind.errorbar(
                plot_pcts, 
                plot_median_rel_mse, 
                yerr=yerr_asymmetric,
                marker='s', 
                linestyle=graphics.client.linestyle,
                color=s_color,
                linewidth=graphics.lines.linewidth,
                capsize=graphics.lines.capsize,
                capthick=graphics.lines.capthick,
                label=graphics.dropout.label_median
            )
            
            ax_ind.set_xlabel(graphics.labels.x.trees_preserved, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
            ax_ind.set_ylabel(graphics.labels.y.dropout_relative_mse, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
            ax_ind.set_xticks(plot_pcts)
            ax_ind.set_xlim(105, 5)
            ax_ind.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
            ax_ind.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
            ax_ind.legend()
            
            plt.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            filename_ind = output_dir / f"client_dropout_analysis_{epsilon}_{strategy}.pdf"
            fig_ind.savefig(filename_ind, bbox_inches='tight')
            plt.close(fig_ind)
            logger.info(f"Gráfico individual salvo: {filename_ind.name}")

            # --- 3. Adicionando os mesmos dados no Gráfico Combinado ---
            ax_combined.errorbar(
                plot_pcts, 
                plot_median_rel_mse, 
                yerr=yerr_asymmetric,
                marker='s', 
                linestyle=graphics.client.linestyle,
                color=s_color,
                linewidth=graphics.lines.linewidth,
                capsize=graphics.lines.capsize,
                capthick=graphics.lines.capthick,
                label=s_label
            )

        # --- 4. Finalizando e Salvando o Gráfico Combinado do Epsilon ---
        ax_combined.set_xlabel(graphics.labels.x.trees_preserved, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
        ax_combined.set_ylabel(graphics.labels.y.dropout_relative_mse, fontsize=graphics.label_fontsize, fontweight=graphics.fontweight)
        ax_combined.set_xticks(plot_pcts)
        ax_combined.set_xlim(105, 5)
        ax_combined.tick_params(axis='both', labelsize=graphics.ticks_fontsize)
        ax_combined.grid(True, linestyle=graphics.grid_linestyle, alpha=graphics.grid_alpha)
        ax_combined.legend()
        
        fig_combined.tight_layout()
        filename_combined = output_dir / f"client_dropout_analysis_{epsilon}_ALL_STRATEGIES.pdf"
        fig_combined.savefig(filename_combined, bbox_inches='tight')
        plt.close(fig_combined)
        logger.info(f"Gráfico combinado salvo: {filename_combined.name}")