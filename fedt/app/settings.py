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
    scripts_path: Path
    client_script_path: Path
    dataset_path: Path
    graphics_path: Path

class SequenceConfig(BaseModel):
    number_of_simulations: int
    aggregation_strategies: list[str]

class ClientConfig(BaseModel):
    timeout: int
    debug: bool
    tree_depth: list[int]
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
    debug: bool

class SettingsConfig(BaseModel):
    number_of_jobs: int
    number_of_clients: int
    number_of_rounds: int
    seeds: list[int]
    epsilons: list[float]
    aggregation_strategy: str
    sequence: SequenceConfig
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