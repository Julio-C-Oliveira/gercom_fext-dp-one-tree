import tomllib
from pathlib import Path
from pydantic import BaseModel
import importlib.resources as res

base_path = ""
base_path = Path(base_path).resolve()

config_path = (base_path / "fedt/simulation/sim_config.toml").resolve()

class SimulationConfig(BaseModel):
    seeds: list[int]
    tree_max_depths: list[int]
    epsilons: list[float]
    balancing_coefficients: list[float]
    number_of_simulations: int
    aggregation_strategies: list[str]

class Config(BaseModel):
    simulation: SimulationConfig


with open(config_path, "rb") as file:
    data = tomllib.load(file)
    config = Config(**data)

    simulation = config.simulation
