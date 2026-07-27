import logging

import numpy as np

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
    def threshold_trees(validation_dataset, received_trees: list[DecisionTreeRegressor], threshold_type, threshold_value, threshold_multiplier):
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
            logger.info(f"Fator: {threshold_multiplier}")
            while not selected_trees:
                threshold_value *= threshold_multiplier
                logger.info(f"Threshold Novo: {threshold_value}")
                selected_trees = [received_trees[i] for i in range(received_trees_number) if tree_scores[i] < threshold_value]


        return selected_trees

    @staticmethod
    def merge_trees(
        received_trees: list[DecisionTreeRegressor],
        max_depth_global: int,
        seed: int,
        n_amostras: int = 20000,
    ) -> DecisionTreeRegressor:
        """
        Funde as árvores dos clientes em uma única DecisionTreeRegressor,
        gerando dados sintéticos dentro dos limites de decisão das árvores recebidas
        e usando o ensemble como professor (Knowledge Distillation).

        Parameters
        ----------
        received_trees : list of DecisionTreeRegressor
            Árvores recebidas dos clientes.
        max_depth_global : int
            Profundidade máxima da árvore global resultante.
        seed : int
            Semente para reprodutibilidade da geração de dados sintéticos.
        n_amostras : int, optional
            Número de amostras sintéticas a gerar. Default: 20000.

        Returns
        -------
        arvore_global : DecisionTreeRegressor
            Árvore única treinada por destilação sobre o ensemble dos clientes.
        """
        if not received_trees:
            raise ValueError("A lista de árvores recebidas está vazia.")

        num_features = received_trees[0].n_features_in_

        feature_mins = np.full(num_features, np.inf)
        feature_maxs = np.full(num_features, -np.inf)

        # 1. Analisar as fronteiras das features nas árvores
        for tree in received_trees:
            tree_structure = tree.tree_
            features_usadas = tree_structure.feature
            thresholds = tree_structure.threshold

            for feat, thresh in zip(features_usadas, thresholds):
                if feat != -2:  # -2 significa nó folha
                    feature_mins[feat] = min(feature_mins[feat], thresh)
                    feature_maxs[feat] = max(feature_maxs[feat], thresh)

        # 2. Corrigir eventuais features não utilizadas para um range padrão
        for i in range(num_features):
            if np.isinf(feature_mins[i]):
                feature_mins[i], feature_maxs[i] = 0.0, 1.0
            elif feature_mins[i] == feature_maxs[i]:
                feature_mins[i] -= 1.0
                feature_maxs[i] += 1.0

        # 3. Gerar malha de dados sintéticos para amostragem
        np.random.seed(seed)
        X_synth = np.random.uniform(
            low=feature_mins,
            high=feature_maxs,
            size=(n_amostras, num_features)
        )

        # 4. Avaliar com o Ensemble (Professor)
        y_synth = np.mean([tree.predict(X_synth) for tree in received_trees], axis=0)

        # 5. Treinar a Árvore Única (Aluno)
        arvore_global = DecisionTreeRegressor(
            max_depth=max_depth_global,
            random_state=seed
        )
        arvore_global.fit(X_synth, y_synth)

        return arvore_global