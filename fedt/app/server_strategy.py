from fedt.app import server_utils

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