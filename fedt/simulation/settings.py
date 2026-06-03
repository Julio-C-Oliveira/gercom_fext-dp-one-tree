import tomllib
from pathlib import Path
from pydantic import BaseModel
import importlib.resources as res

base_path = ""
base_path = Path(base_path).resolve()

config_path = (base_path / "fedt/simulation/sim_config.toml").resolve()

class EpsilonSetting(BaseModel):
    epsilon: float
    balancing_coefficient: float  
    threshold_value: float
    threshold_type: str
    threshold_multiplier: float

class SimulationConfig(BaseModel):
    seeds: list[int]
    tree_max_depths: list[int]
    number_of_simulations: int
    aggregation_strategies: list[str]
    number_of_clients_for_test: int
    start_ID_for_clients: int
    dirichlet_alpha: float
    number_of_bins_for_dirichlet: int
    epsilon_settings: list[EpsilonSetting]

class Config(BaseModel):
    simulation: SimulationConfig


with open(config_path, "rb") as file:
    data = tomllib.load(file)
    config = Config(**data)

    simulation = config.simulation
