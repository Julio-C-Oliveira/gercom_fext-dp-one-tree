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