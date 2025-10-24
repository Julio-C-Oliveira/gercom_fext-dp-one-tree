from app.settings import dataset_path, percentage_value_of_samples_per_client, train_test_split_size, validate_dataset_size, number_of_clients

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset():
    energy_dataset = pd.read_csv(dataset_path)
    columns_for_training = []
    temperature_columns = [f"T{i}" for i in range(1, 10)]
    humidity_columns = [f"RH_{i}" for i in range(1, 10)]

    for temperature in temperature_columns:
        columns_for_training.append(temperature)
            
    for humidity in humidity_columns:
        columns_for_training.append(humidity)

    columns_for_training.append("T_out")
    columns_for_training.append("RH_out")
    columns_for_training.append("Press_mm_hg")
    columns_for_training.append("Visibility")

    X = energy_data_complete[columns_for_training]
    y = energy_data_complete["Appliances"]

    dataset_size = len(X)

    return X, y, dataset_size

def load_client_dataset(seed):
    rng = np.random.default_rng(seed=random_state)

    X, y, dataset_size = load_dataset()

    number_of_samples = int((dataset_size*percentage_value_of_samples_per_client)/100)

    indexs = rng.choice(
        X.shape[0], 
        size=number_of_samples, 
        replace=False
    )
    X = X.iloc[indexs]
    y = y.iloc[indexs]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=train_test_split_size, 
        random_state=random_state
    )
    return X_train, y_train, X_test, y_test

def load_server_initialization_dataset():
    X, y, _ = load_dataset()
    return X[:2], y[:2]

def load_server_validation_dataset():
    X, y, _ = load_dataset()
    return X[-validate_dataset_size:], y[-validate_dataset_size:]

def load_all_datasets(seed):
    return [load_client_dataset(int(seed/(i+1))) for i in range(number_of_clients)], load_server_initialization_dataset(), load_server_validation_dataset()

# TO-DO:
# - Create a better method to define the clients seed.