from fedt.app.settings import settings

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_threshold_and_evaluate_function():
    match threshold_type:
        case "pearson":
            threshold = settings.server.pearson_threshold
            evaluate_function = pearsonr
        case "mse":
            threshold = settings.server.mean_squared_error_threhsold
            evaluate_function = mean_squared_error
        case _:
            threshold = settings.server.pearson_threshold
            evaluate_function = pearsonr

    return threshold, evaluate_function