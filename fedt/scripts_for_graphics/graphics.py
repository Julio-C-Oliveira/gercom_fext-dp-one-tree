from fedt.scripts_for_graphics.simulation_graphics import plot_simulation_graphics
from fedt.scripts_for_graphics.membership_inference_attack_graphics import plot_membership_inference_attack_graphics
from fedt.scripts_for_graphics.data_reconstruction_attack_graphics import plot_data_reconstruction_attack_graphics
from fedt.scripts_for_graphics.shap_graphics import plot_shap_analysis_graphics
from fedt.scripts_for_graphics.ensemble_graphics import plot_ensemble_analysis_graphics

if __name__ == "__main__":
    # plot_simulation_graphics()
    # plot_membership_inference_attack_graphics()
    # plot_data_reconstruction_attack_graphics()
    # plot_shap_analysis_graphics()
    plot_ensemble_analysis_graphics()

    print("Gráficos gerados e salvos com sucesso!")