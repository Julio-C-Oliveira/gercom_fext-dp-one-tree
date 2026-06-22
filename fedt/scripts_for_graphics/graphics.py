from fedt.scripts_for_graphics.simulation_graphics import plot_simulation_graphics
from fedt.scripts_for_graphics.membership_inference_attack_graphics import plot_membership_inference_attack_graphics
from fedt.scripts_for_graphics.data_reconstruction_attack_graphics import plot_data_reconstruction_attack_graphics

def plot_shap_graphics():
    pass

def plot_ensemble_analysis_graphics():
    pass

if __name__ == "__main__":
    plot_simulation_graphics()
    plot_membership_inference_attack_graphics()
    plot_data_reconstruction_attack_graphics()

    print("Gráficos gerados e salvos com sucesso!")