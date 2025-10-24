from app.settings import server_tree_depth

from app import server_utils

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class Strategy():
    @staticmethod
    def all_trees(received_trees: list[DecisionTreeRegressor]):
        """
        Returns all trees sent by clients.

        Parameters
        ----------
        received_trees : list of trees sent by clients. 

        Returns
        -------
        received_trees : list of trees sent by clients. 
        """
        return received_trees

    @staticmethod
    def threshold_trees(validation_dataset, received_trees: list[DecisionTreeRegressor]):
        """
        Returns the trees that cross the threshold.

        Parameters
        ---------- 
        validation_dataset : 

        received_trees : list of trees sent by clients. 

        threshold : a float that split the trees selecting the best models.

        Returns
        -------
        selected_trees : the best trees according to the server.
        """
        X_validate, y_validate = validation_dataset
        best_trees = []

        received_trees_number = len(received_trees)

        threshold, evaluate_function = server_utils.get_threshold_and_evaluate_function()

        map_function = lambda tree: evaluate_function(y_validate, tree.predict(X_validate))

        tree_scores = list(map(map_function, received_trees))

        selected_trees = [received_trees[i] for i in range(received_trees_number) if tree_scores[i] > threshold]

        return selected_trees

class Server():
    def __init__(self, inicialization_dataset, validation_dataset, epsilon: float):
        self.global_model = RandomForestRegressor(
            n_estimators=2,
            max_depth=server_tree_depth
        )

        X_train, y_train = inicialization_dataset
        self.validation_dataset = validation_dataset

        self.global_model.fit(
            X_train,
            y_train,
            epsilon=epsilon
        )

    def aggregate_trees(received_trees: list[DecisionTreeRegressor], aggregation_strategy: str):
        match aggregation_strategy:
            case "all_trees":
                self.global_model.estimators_ = Strategy.all_trees(received_trees)
            case "threshold_trees": 
                self.global_model.estimators_ = Strategy.threshold_trees(self.validation_dataset, received_trees)
            case _:
                self.global_model.estimators_ = Strategy.all_trees(received_trees)


"""
Return boolean mask denoting if there are missing values for each feature.

This method also ensures that X is finite.

Parameter
---------
X : array-like of shape (n_samples, n_features), dtype=DOUBLE
    Input data.

estimator_name : str or None, default=None
    Name to use when raising an error. Defaults to the class name.

Returns
-------
missing_values_in_feature_mask : ndarray of shape (n_features,), or None
    Missing value mask. If missing values are not supported or there
    are no missing values, return None.
"""