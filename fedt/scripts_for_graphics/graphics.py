from fedt.scripts_for_graphics.simulation_graphics import plot_simulation_graphics
from fedt.scripts_for_graphics.membership_inference_attack_graphics import plot_membership_inference_attack_graphics
from fedt.scripts_for_graphics.data_reconstruction_attack_graphics import plot_data_reconstruction_attack_graphics
from fedt.scripts_for_graphics.shap_graphics import plot_shap_analysis_graphics
from fedt.scripts_for_graphics.client_dropout_analysis_graphics import plot_client_dropout_analysis_graphics
from fedt.scripts_for_graphics.explainability_eval_graphics import plot_explainability_eval_graphics

import logging
from fedt.app import utils

if __name__ == "__main__":
    log_level = logging.DEBUG if True else logging.INFO
    logger = utils.setup_logger(
        name=f"GRAPHICS",
        log_file=f"graphics.log",
        level=log_level
    )

    logger.info("Iniciando a geração dos gráficos.")

    graphics = [
        ("Simulação Base", plot_simulation_graphics),
        ("MIA", plot_membership_inference_attack_graphics),
        ("Reconstruct Attack", plot_data_reconstruction_attack_graphics),
        # ("SHAP", plot_shap_analysis_graphics),
        # ("XAI", plot_explainability_eval_graphics),
        # ("Dropout", plot_client_dropout_analysis_graphics)
    ]

    for name, generate_function in graphics:
        try:
            logger.info(f"Processando gráficos de {name}...")
            generate_function()
            logger.info(f"Gráficos de {name} concluídos.")
        except Exception as e:
            logger.error(f"Falha ao gerar gráficos de {name}: {e}", exc_info=True)

    logger.info("Gráficos gerados e salvos com sucesso!")