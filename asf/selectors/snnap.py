import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Optional

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.selectors.abstract_selector import AbstractSelector
from functools import partial


class SNNAP(AbstractSelector):
    """
    SATzilla-like Nearest Neighbor Algorithm Portfolio (SNNAP).
    
    This selector uses k-nearest neighbors to select algorithms based on
    the performance of algorithms on similar instances in the feature space.
    
    Attributes:
        k (int): Number of nearest neighbors to consider.
        weight_strategy (str): Strategy for weighting neighbors ('uniform', 'distance').
        algorithm_selection_strategy (str): Strategy for selecting algorithm from neighbors 
                                          ('voting', 'best_performance').
        scaler (StandardScaler): Scaler for normalizing features.
        nn_model (NearestNeighbors): Nearest neighbors model.
        training_features (pd.DataFrame): Training feature data.
        training_performance (pd.DataFrame): Training performance data.
    """

    PREFIX = "snnap"

    def __init__(
        self,
        k: int = 5,
        weight_strategy: str = "distance",
        algorithm_selection_strategy: str = "voting",
        **kwargs
    ):
        """
        Initialize SNNAP selector.

        Args:
            k (int): Number of nearest neighbors to consider. Defaults to 5.
            weight_strategy (str): Strategy for weighting neighbors. 
                                 Options: 'uniform', 'distance'. Defaults to 'distance'.
            algorithm_selection_strategy (str): Strategy for selecting algorithm from neighbors.
                                               Options: 'voting', 'best_performance'. 
                                               Defaults to 'voting'.
            **kwargs: Additional arguments for AbstractSelector.
        """
        super().__init__(**kwargs)
        self.k = k
        self.weight_strategy = weight_strategy
        self.algorithm_selection_strategy = algorithm_selection_strategy
        
        # Initialize models and data storage
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.training_features: Optional[pd.DataFrame] = None
        self.training_performance: Optional[pd.DataFrame] = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the SNNAP selector with training data.

        Args:
            features (pd.DataFrame): Training feature data.
            performance (pd.DataFrame): Training performance data.
        """
        # Store training data
        self.training_features = features.copy()
        self.training_performance = performance.copy()
        
        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features)
        
        # Adjust k if it's larger than the number of training samples
        effective_k = min(self.k, len(features))
        
        # Fit nearest neighbors model with adjusted k
        self.nn_model = NearestNeighbors(n_neighbors=effective_k, metric='euclidean')
        self.nn_model.fit(features_scaled)

    def _predict(
        self, features: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
        """
        Predict the best algorithm for each instance using k-nearest neighbors.

        Args:
            features (pd.DataFrame): Feature data for instances to predict.

        Returns:
            Dict[str, List[Tuple[str, Union[int, float]]]]: Dictionary mapping instance names 
                                                           to selected algorithm and budget.
        """
        if self.training_features is None or self.training_performance is None:
            raise ValueError("Model must be fitted before making predictions")

        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Find k-nearest neighbors
        distances, indices = self.nn_model.kneighbors(features_scaled)
        
        predictions = {}
        
        for i, instance_name in enumerate(features.index):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Get performance data for neighbors
            neighbor_performance = self.training_performance.iloc[neighbor_indices]
            
            # Select algorithm based on strategy
            if self.algorithm_selection_strategy == "voting":
                selected_algorithm = self._voting_strategy(
                    neighbor_performance, neighbor_distances
                )
            elif self.algorithm_selection_strategy == "best_performance":
                selected_algorithm = self._best_performance_strategy(
                    neighbor_performance, neighbor_distances
                )
            else:
                raise ValueError(f"Unknown algorithm selection strategy: {self.algorithm_selection_strategy}")
            
            predictions[instance_name] = [(selected_algorithm, self.budget)]
        
        return predictions

    def _voting_strategy(
        self, neighbor_performance: pd.DataFrame, neighbor_distances: np.ndarray
    ) -> str:
        """
        Select algorithm using voting strategy.
        Each neighbor votes for the algorithm that performed best on it.

        Args:
            neighbor_performance (pd.DataFrame): Performance data for neighbors.
            neighbor_distances (np.ndarray): Distances to neighbors.

        Returns:
            str: Selected algorithm name.
        """
        # Find best algorithm for each neighbor
        if self.maximize:
            best_algorithms = neighbor_performance.idxmax(axis=1)
        else:
            best_algorithms = neighbor_performance.idxmin(axis=1)
        
        # Calculate weights based on strategy
        if self.weight_strategy == "uniform":
            weights = np.ones(len(neighbor_distances))
        elif self.weight_strategy == "distance":
            # Closer neighbors have higher weights (inverse distance weighting)
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (neighbor_distances + 1e-10)
        else:
            raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")
        
        # Count weighted votes for each algorithm
        algorithm_votes = {}
        for algo, weight in zip(best_algorithms, weights):
            algorithm_votes[algo] = algorithm_votes.get(algo, 0) + weight
        
        # Return algorithm with most votes
        return max(algorithm_votes.items(), key=lambda x: x[1])[0]

    def _best_performance_strategy(
        self, neighbor_performance: pd.DataFrame, neighbor_distances: np.ndarray
    ) -> str:
        """
        Select algorithm based on best overall performance among neighbors.

        Args:
            neighbor_performance (pd.DataFrame): Performance data for neighbors.
            neighbor_distances (np.ndarray): Distances to neighbors.

        Returns:
            str: Selected algorithm name.
        """
        # Calculate weights based on strategy
        if self.weight_strategy == "uniform":
            weights = np.ones(len(neighbor_distances))
        elif self.weight_strategy == "distance":
            # Closer neighbors have higher weights
            weights = 1.0 / (neighbor_distances + 1e-10)
        else:
            raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")
        
        # Calculate weighted average performance for each algorithm
        weighted_performance = {}
        for algorithm in self.algorithms:
            if self.maximize:
                # For maximization, higher values are better
                weighted_avg = np.average(neighbor_performance[algorithm], weights=weights)
            else:
                # For minimization, lower values are better
                weighted_avg = np.average(neighbor_performance[algorithm], weights=weights)
            weighted_performance[algorithm] = weighted_avg
        
        # Return algorithm with best weighted average performance
        if self.maximize:
            return max(weighted_performance.items(), key=lambda x: x[1])[0]
        else:
            return min(weighted_performance.items(), key=lambda x: x[1])[0]

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            cs_transform: Optional[Dict[str, dict]] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
            **kwargs,
        ) -> Tuple[ConfigurationSpace, Dict[str, dict]]:
            """
            Get the configuration space for SNNAP.

            Args:
                cs (Optional[ConfigurationSpace]): Existing configuration space.
                cs_transform (Optional[Dict[str, dict]]): Transform dictionary.
                pre_prefix (str): Prefix for parameter names.
                parent_param (Optional[Hyperparameter]): Parent parameter for conditions.
                parent_value (Optional[str]): Parent parameter value for conditions.
                **kwargs: Additional keyword arguments.

            Returns:
                Tuple[ConfigurationSpace, Dict[str, dict]]: Configuration space and transform dict.
            """
            if cs is None:
                cs = ConfigurationSpace()

            if cs_transform is None:
                cs_transform = dict()

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SNNAP.PREFIX}"
            else:
                prefix = SNNAP.PREFIX

            # Define parameters
            k_param = Integer(
                name=f"{prefix}:k",
                bounds=(1, 20),
                default=5,
            )

            weight_strategy_param = Categorical(
                name=f"{prefix}:weight_strategy",
                items=["uniform", "distance"],
                default="distance",
            )

            algorithm_selection_strategy_param = Categorical(
                name=f"{prefix}:algorithm_selection_strategy",
                items=["voting", "best_performance"],
                default="voting",
            )

            params = [k_param, weight_strategy_param, algorithm_selection_strategy_param]

            # Add conditions if parent parameter exists
            if parent_param is not None:
                conditions = [
                    EqualsCondition(
                        child=param,
                        parent=parent_param,
                        value=parent_value,
                    )
                    for param in params
                ]
            else:
                conditions = []

            cs.add(params + conditions)

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: Dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> "SNNAP":
            """
            Create SNNAP instance from configuration.

            Args:
                configuration (Configuration): Configuration object.
                cs_transform (Dict[str, dict]): Transform dictionary.
                pre_prefix (str): Prefix for parameter names.
                **kwargs: Additional keyword arguments.

            Returns:
                SNNAP: Configured SNNAP instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SNNAP.PREFIX}"
            else:
                prefix = SNNAP.PREFIX

            k = configuration[f"{prefix}:k"]
            weight_strategy = configuration[f"{prefix}:weight_strategy"]
            algorithm_selection_strategy = configuration[f"{prefix}:algorithm_selection_strategy"]

            return SNNAP(
                k=k,
                weight_strategy=weight_strategy,
                algorithm_selection_strategy=algorithm_selection_strategy,
                **kwargs,
            )