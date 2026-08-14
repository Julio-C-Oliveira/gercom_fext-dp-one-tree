from fedt.app.settings import settings

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_threshold_and_evaluate_function(threshold_type):
    match threshold_type:
        case "pearson":
            evaluate_function = pearsonr
        case "mse":
            evaluate_function = mean_squared_error
        case _:
            evaluate_function = mean_squared_error

    return evaluate_function

from fedt.app.utils import load_house_client, get_final_seed
import numpy as np
import pandas as pd
from fedt.app.settings import dataset

def build_global_test_data(number_of_clients, seed):
    "Não use isso em qualquer ponto da agregação do servidor, esses dados servem apenas para o teste de cross validation."

    X_test_list = []
    y_test_list = []

    for client_id in range(number_of_clients):
        client_seed = get_final_seed(client_id, seed)
        
        _, _, X_test, y_test = load_house_client(
            seed=client_seed,
            alpha=settings.client.dirichlet_alpha,
            bins=settings.client.number_of_bins_for_dirichlet,
            percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client
        )
        
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    X_global_test = pd.concat(X_test_list, axis=0, ignore_index=True)
    y_global_test = pd.concat(y_test_list, axis=0, ignore_index=True)
    return X_global_test, y_global_test

def cross_validation_test(model, global_test_data):
    "Unicamente para fins de teste de generalização, obviamente isso não seria válido em produção."

    X_global_test, y_global_test = global_test_data
    y_pred = model.predict(X_global_test)

    mse = mean_squared_error(y_global_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "mse": mse,
        "rmse": rmse
    }