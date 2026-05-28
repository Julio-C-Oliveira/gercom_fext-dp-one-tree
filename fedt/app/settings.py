import tomllib
from pathlib import Path
from pydantic import BaseModel
import importlib.resources as res

base_path = ""
base_path = Path(base_path).resolve()

config_path = (base_path / "fedt/app/app_config.toml").resolve()

class PathConfig(BaseModel):
    base_path: Path
    results_folder: Path
    final_results_folder: Path
    logs_folder: Path
    dataset_path: Path
    graphics_path: Path

class DifferentialPrivacyConfig(BaseModel):
    splitter: str
    global_max_target: float
    global_min_target: float
    tree_max_depth: int
    epsilon: float
    balancing_coefficient: float

class ClientConfig(BaseModel):
    timeout: int
    evaluate_type: str

class ServerConfig(BaseModel):
    IP: str
    port: str
    pearson_threshold: float
    mean_squared_error_threhsold: float
    threshold_type: str
    validate_dataset_size: int
    print_every_trees_sent: int
    timeout: int

class SettingsConfig(BaseModel):
    number_of_jobs: int
    number_of_clients: int
    number_of_rounds: int
    seed: int
    aggregation_strategy: str
    differential_privacy: DifferentialPrivacyConfig
    client: ClientConfig
    server: ServerConfig

class DatasetConfig(BaseModel):
    train_test_split_size: float
    percentage_value_of_samples_per_client: int

class ScriptsConfig(BaseModel):
    network_interface: str

class Config(BaseModel):
    paths: PathConfig
    settings: SettingsConfig
    dataset: DatasetConfig
    scripts: ScriptsConfig

with open(config_path, "rb") as file:
    data = tomllib.load(file)
    config = Config(**data)

    paths = config.paths
    settings = config.settings
    dataset = config.dataset
    scripts = config.scripts