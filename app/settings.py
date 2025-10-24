import tomllib
from pathlib import Path

pyproject_path = "/home/julio-vm/Gercom/Centralized/FEXT-DP_ONE_TREE/pyproject.toml"

with open(pyproject_path, "rb") as file:
    config = tomllib.load(file)

results_folder = config["paths"]["results_folder"]
dataset_path = config["paths"]["dataset_path"]

number_of_jobs = config["settings"]["number_of_jobs"]
number_of_clients = config["settings"]["number_of_clients"]
number_of_rounds = config["settings"]["number_of_rounds"]
seeds = config["settings"]["seeds"]
epsilons = config["settings"]["epsilons"]
aggregation_strategies = config["settings"]["aggregation_strategies"]

client_tree_depth = config["settings"]["client"]["tree_depth"]
evaluate_type = config["settings"]["client"]["evaluate_type"]

server_tree_depth = config["settings"]["server"]["tree_depth"]
pearson_threshold = config["settings"]["server"]["pearson_threshold"]
mse_threshold = config["settings"]["server"]["mean_squared_error_threhsold"]
threshold_type = config["settings"]["server"]["threshold_type"]
validate_dataset_size = config["settings"]["server"]["validate_dataset_size"]

train_test_split_size = config["dataset"]["train_test_split_size"]
percentage_value_of_samples_per_client = config["dataset"]["percentage_value_of_samples_per_client"]