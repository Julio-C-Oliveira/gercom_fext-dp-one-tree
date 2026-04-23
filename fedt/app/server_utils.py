from fedt.app.settings import pearson_threshold, mse_threshold, threshold_type

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def get_threshold_and_evaluate_function():
    match threshold_type:
        case "pearson":
            threshold = pearson_threshold
            evaluate_function = pearsonr
        case "mse":
            threshold = mse_threshold
            evaluate_function = mean_squared_error
        case _:
            threshold = pearson_threshold
            evaluate_function = pearsonr

    return threshold, evaluate_function