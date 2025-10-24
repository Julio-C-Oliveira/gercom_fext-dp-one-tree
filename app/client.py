from app.settings import client_tree_depth, evaluate_type

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

class Client():
    def __init__(dataset, epsilon: float):
        self.X_train, self.y_train, self.X_test, self.y_test = dataset

        self.model = DecisionTreeRegressor(
            max_depth=client_tree_depth
        )
        self.model.fit(
            self.X_train,
            self.y_train,
            epsilon=epsilon
        )

    def choose_model(self, global_model):
        local_model_predictions = self.model.predict(self.X_test)
        global_model_predictions = global_model.predict(self.X_test)

        metrics = evaluate(local_model_predictions)
        global_metrics = evaluate(global_model_predictions)

        if global_metrics[evaluate_type] < metrics[evaluate_type]: 
            self.model = global_model
            metrics = global_metrics

        return metrics

    def evaluate(self, predicts):
        metrics = {}

        metrics["mse"] = mean_squared_error(self.y_test, predicts)
        metrics["pearson"] = pearsonr(self.y_test, predicts)[0] # Somente a correlação me interessa, que é o primeiro valor.

        return metrics