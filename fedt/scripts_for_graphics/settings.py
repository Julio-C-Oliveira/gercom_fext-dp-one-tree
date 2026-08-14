import tomllib
from pathlib import Path
from pydantic import BaseModel

base_path = ""
base_path = Path(base_path).resolve()

config_path = (base_path / "fedt/scripts_for_graphics/graphics_config.toml").resolve()


class LinesConfig(BaseModel):
    linewidth: float
    capsize: float
    capthick: float
    server_linestyle: str


class BoxplotConfig(BaseModel):
    box_facecolor: str
    box_color: str
    median_color: str
    median_linewidth: float


class StyleConfig(BaseModel):
    color: str
    marker: str
    linestyle: str
    label: str


class StrategyConfig(BaseModel):
    color: str
    marker: str
    label: str


class ServerConfig(BaseModel):
    label: str
    label_combined_format: str


class DropoutConfig(BaseModel):
    label_median: str


class LabelsXConfig(BaseModel):
    privacy_level: str
    trees_preserved: str


class LabelsEpsilonConfig(BaseModel):
    no_diff_privacy: str


class LabelsYConfig(BaseModel):
    # Simulação Base
    initial_mse: str
    initial_rmse: str
    final_mse: str
    final_rmse: str
    cross_validation_mse: str
    cross_validation_rmse: str
    # Client Dropout
    dropout_relative_mse: str
    # Ataques
    dra_mse: str
    dra_rmse: str
    mia_accuracy: str
    mia_auc: str
    # Explicabilidade (XAI)
    hoyer_sparsity: str
    gini_index: str
    mae_fidelity: str
    spearman_rank_corr: str
    jaccard_top3: str
    jaccard_top5: str
    cosine_distance: str
    local_sensitivity: str
    gap_cosine_distance: str
    gap_jaccard_top3: str
    gap_spearman_rank_corr: str


class LabelsConfig(BaseModel):
    x: LabelsXConfig
    epsilon: LabelsEpsilonConfig
    y: LabelsYConfig


class GraphicsConfig(BaseModel):
    wide_figsize: list
    normal_figsize: list
    label_fontsize: float
    ticks_fontsize: float
    legend_fontsize: float
    fontweight: str
    grid_linestyle: str
    grid_alpha: float
    remove_outliers: str
    lines: LinesConfig
    boxplot: BoxplotConfig
    client: StyleConfig
    server: ServerConfig
    dropout: DropoutConfig
    sbdt: StyleConfig
    strategies: dict[str, StrategyConfig]
    labels: LabelsConfig


class Config(BaseModel):
    graphics: GraphicsConfig


with open(config_path, "rb") as file:
    data = tomllib.load(file)
    config = Config(**data)

    graphics = config.graphics