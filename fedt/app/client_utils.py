import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from fedt.app import utils

import warnings
from scipy.stats import ConstantInputWarning

from fedt.app.settings import settings

warnings.filterwarnings("ignore", category=ConstantInputWarning)

class Client():

    def __init__(self, trees_by_client: int, dataset, ID, seed, epsilon) -> None:
        self.X_train, self.y_train, self.X_test, self.y_test = dataset

        self.local_model = DecisionTreeRegressor(
            max_depth=settings.differential_privacy.tree_max_depth,
            splitter=settings.differential_privacy.splitter,
            random_state=seed
        )
        self.local_model.fit(
            self.X_train,
            self.y_train,
            global_max_target=settings.differential_privacy.global_max_target,
            global_min_target=settings.differential_privacy.global_min_target,
            epsilon_global_budget=epsilon,
            balancing_coefficient=settings.differential_privacy.balancing_coefficient
        )

        self.ID = ID

    def choose_model(self, global_model):
        local_model_predictions = self.local_model.predict(self.X_test)
        global_model_predictions = global_model.predict(self.X_test)

        metrics = evaluate(local_model_predictions)
        global_metrics = evaluate(global_model_predictions)

        if global_metrics[settings.client.evaluate_type] < metrics[settings.client.evaluate_type]: 
            self.model = global_model
            metrics = global_metrics

        return metrics

    def evaluate(self, predicts):
        metrics = {}

        metrics["mse"] = mean_squared_error(self.y_test, predicts)
        metrics["pearson"] = pearsonr(self.y_test, predicts)[0]

        return metrics

    def evaluate_inference_time(self, number_of_samples):
        self.local_model.predict(self.X_test[-number_of_samples:])