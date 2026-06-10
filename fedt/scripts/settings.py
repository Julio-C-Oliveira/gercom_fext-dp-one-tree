import tomllib
from pathlib import Path
from pydantic import BaseModel
import importlib.resources as res

base_path = ""
base_path = Path(base_path).resolve()

config_path = (base_path / "fedt/scripts/scripts_config.toml").resolve()

class GraphicsConfig(BaseModel):
    fontsize: float
    fontweight: str
    ticks_fontsize: float
    normal_figsize: list
    grid_linestyle: str
    grid_alpha: float
    remove_outliers: str

class Config(BaseModel):
    graphics: GraphicsConfig

with open(config_path, "rb") as file:
    data = tomllib.load(file)
    config = Config(**data)

    graphics = config.graphics