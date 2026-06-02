import logging

from fedt.app import server_utils

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger("SERVER")

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
    def threshold_trees(validation_dataset, received_trees: list[DecisionTreeRegressor], threshold_type, threshold_value):
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

        evaluate_function = server_utils.get_threshold_and_evaluate_function(threshold_type)

        map_function = lambda tree: evaluate_function(y_validate, tree.predict(X_validate))

        tree_scores = list(map(map_function, received_trees))

        selected_trees = [received_trees[i] for i in range(received_trees_number) if tree_scores[i] < threshold_value] # Tenho que tornar essa condição genérica. 

        if not selected_trees:
            logger.warning("Nenhuma árvore foi selecionada. Iniciando adaptação.")
            logger.info(f"Threshold Atual: {threshold_value}")
            while not selected_trees:
                threshold_value *= 1.001
                logger.info(f"Threshold Novo: {threshold_value}")
                selected_trees = [received_trees[i] for i in range(received_trees_number) if tree_scores[i] < threshold_value]


        return selected_trees