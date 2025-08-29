import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Union, Optional

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Float,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.selectors.abstract_selector import AbstractSelector
from functools import partial


class ISAC(AbstractSelector):
    """
    Instance-Specific Algorithm Configuration (ISAC).
    
    This selector uses instance features to predict instance-specific algorithm
    configurations or selections. It can work in two modes:
    1. Clustering-based: Groups similar instances and learns best algorithm for each cluster
    2. Regression-based: Uses features to directly predict algorithm performance
    
    Attributes:
        mode (str): Operating mode ('clustering' or 'regression').
        n_clusters (int): Number of clusters for clustering mode.
        distance_threshold (float): Distance threshold for assigning to clusters.
        fallback_strategy (str): Strategy when no good assignment is found.
        scaler (StandardScaler): Feature scaler.
        clustering_model (KMeans): Clustering model for clustering mode.
        nn_model (NearestNeighbors): Nearest neighbors model for regression mode.
        cluster_algorithms (Dict): Best algorithm for each cluster.
        training_features (pd.DataFrame): Training feature data.
        training_performance (pd.DataFrame): Training performance data.
    """

    PREFIX = "isac"

    def __init__(
        self,
        mode: str = "clustering",
        n_clusters: int = 10,
        distance_threshold: float = 1.0,
        fallback_strategy: str = "best_overall",
        **kwargs
    ):
        """
        Initialize ISAC selector.

        Args:
            mode (str): Operating mode. Options: 'clustering', 'regression'. 
                       Defaults to 'clustering'.
            n_clusters (int): Number of clusters for clustering mode. Defaults to 10.
            distance_threshold (float): Distance threshold for cluster assignment. 
                                      Defaults to 1.0.
            fallback_strategy (str): Strategy when no good assignment found.
                                   Options: 'best_overall', 'random'. 
                                   Defaults to 'best_overall'.
            **kwargs: Additional arguments for AbstractSelector.
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.fallback_strategy = fallback_strategy
        
        # Initialize models and data storage
        self.scaler = StandardScaler()
        self.clustering_model: Optional[KMeans] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.cluster_algorithms: Dict[int, str] = {}
        self.overall_best_algorithm: Optional[str] = None
        self.training_features: Optional[pd.DataFrame] = None
        self.training_performance: Optional[pd.DataFrame] = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the ISAC selector with training data.

        Args:
            features (pd.DataFrame): Training feature data.
            performance (pd.DataFrame): Training performance data.
        """
        # Store training data
        self.training_features = features.copy()
        self.training_performance = performance.copy()
        
        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features)
        
        # Find overall best algorithm for fallback
        if self.maximize:
            overall_performance = performance.mean()
            self.overall_best_algorithm = overall_performance.idxmax()
        else:
            overall_performance = performance.mean()
            self.overall_best_algorithm = overall_performance.idxmin()
        
        if self.mode == "clustering":
            self._fit_clustering_mode(features_scaled, performance)
        elif self.mode == "regression":
            self._fit_regression_mode(features_scaled, performance)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _fit_clustering_mode(
        self, features_scaled: np.ndarray, performance: pd.DataFrame
    ) -> None:
        """
        Fit ISAC in clustering mode.

        Args:
            features_scaled (np.ndarray): Scaled feature data.
            performance (pd.DataFrame): Performance data.
        """
        # Fit clustering model
        self.clustering_model = KMeans(
            n_clusters=min(self.n_clusters, len(features_scaled)),
            random_state=42,
            n_init=10
        )
        cluster_labels = self.clustering_model.fit_predict(features_scaled)
        
        # For each cluster, find the best algorithm
        for cluster_id in range(self.clustering_model.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_performance = performance.loc[cluster_mask]
            
            if len(cluster_performance) > 0:
                if self.maximize:
                    cluster_avg_performance = cluster_performance.mean()
                    best_algorithm = cluster_avg_performance.idxmax()
                else:
                    cluster_avg_performance = cluster_performance.mean()
                    best_algorithm = cluster_avg_performance.idxmin()
                
                self.cluster_algorithms[cluster_id] = best_algorithm

    def _fit_regression_mode(
        self, features_scaled: np.ndarray, performance: pd.DataFrame
    ) -> None:
        """
        Fit ISAC in regression mode.

        Args:
            features_scaled (np.ndarray): Scaled feature data.
            performance (pd.DataFrame): Performance data.
        """
        # In regression mode, we use k-nearest neighbors to predict performance
        # Adjust k if it's larger than the number of training samples
        effective_k = min(5, len(features_scaled))
        self.nn_model = NearestNeighbors(n_neighbors=effective_k, metric='euclidean')
        self.nn_model.fit(features_scaled)

    def _predict(
        self, features: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
        """
        Predict the best algorithm for each instance.

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
        
        predictions = {}
        
        for i, instance_name in enumerate(features.index):
            instance_features = features_scaled[i:i+1]
            
            if self.mode == "clustering":
                selected_algorithm = self._predict_clustering_mode(instance_features)
            elif self.mode == "regression":
                selected_algorithm = self._predict_regression_mode(instance_features)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            predictions[instance_name] = [(selected_algorithm, self.budget)]
        
        return predictions

    def _predict_clustering_mode(self, instance_features: np.ndarray) -> str:
        """
        Predict algorithm using clustering mode.

        Args:
            instance_features (np.ndarray): Scaled features for single instance.

        Returns:
            str: Selected algorithm name.
        """
        if self.clustering_model is None:
            return self._get_fallback_algorithm()
        
        # Find closest cluster
        cluster_centers = self.clustering_model.cluster_centers_
        distances = np.linalg.norm(cluster_centers - instance_features, axis=1)
        closest_cluster = np.argmin(distances)
        
        # Check if instance is close enough to assigned cluster
        if distances[closest_cluster] <= self.distance_threshold:
            if closest_cluster in self.cluster_algorithms:
                return self.cluster_algorithms[closest_cluster]
        
        # Fallback if no good cluster assignment
        return self._get_fallback_algorithm()

    def _predict_regression_mode(self, instance_features: np.ndarray) -> str:
        """
        Predict algorithm using regression mode.

        Args:
            instance_features (np.ndarray): Scaled features for single instance.

        Returns:
            str: Selected algorithm name.
        """
        if self.nn_model is None:
            return self._get_fallback_algorithm()
        
        # Find k-nearest neighbors
        distances, indices = self.nn_model.kneighbors(instance_features)
        
        # Get performance data for neighbors
        neighbor_performance = self.training_performance.iloc[indices[0]]
        
        # Calculate weighted average performance for each algorithm
        weights = 1.0 / (distances[0] + 1e-10)  # Inverse distance weighting
        weighted_performance = {}
        
        for algorithm in self.algorithms:
            weighted_avg = np.average(neighbor_performance[algorithm], weights=weights)
            weighted_performance[algorithm] = weighted_avg
        
        # Return algorithm with best weighted average performance
        if self.maximize:
            return max(weighted_performance.items(), key=lambda x: x[1])[0]
        else:
            return min(weighted_performance.items(), key=lambda x: x[1])[0]

    def _get_fallback_algorithm(self) -> str:
        """
        Get fallback algorithm when no good prediction can be made.

        Returns:
            str: Fallback algorithm name.
        """
        if self.fallback_strategy == "best_overall":
            return self.overall_best_algorithm or self.algorithms[0]
        elif self.fallback_strategy == "random":
            return np.random.choice(self.algorithms)
        else:
            raise ValueError(f"Unknown fallback strategy: {self.fallback_strategy}")

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
            Get the configuration space for ISAC.

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
                prefix = f"{pre_prefix}:{ISAC.PREFIX}"
            else:
                prefix = ISAC.PREFIX

            # Define parameters
            mode_param = Categorical(
                name=f"{prefix}:mode",
                items=["clustering", "regression"],
                default="clustering",
            )

            n_clusters_param = Integer(
                name=f"{prefix}:n_clusters",
                bounds=(2, 50),
                default=10,
            )

            distance_threshold_param = Float(
                name=f"{prefix}:distance_threshold",
                bounds=(0.1, 5.0),
                default=1.0,
            )

            fallback_strategy_param = Categorical(
                name=f"{prefix}:fallback_strategy",
                items=["best_overall", "random"],
                default="best_overall",
            )

            params = [mode_param, n_clusters_param, distance_threshold_param, fallback_strategy_param]

            # Add conditions for clustering-specific parameters
            clustering_condition = EqualsCondition(
                child=n_clusters_param,
                parent=mode_param,
                value="clustering",
            )

            clustering_condition2 = EqualsCondition(
                child=distance_threshold_param,
                parent=mode_param,
                value="clustering",
            )

            conditions = [clustering_condition, clustering_condition2]

            # Add conditions if parent parameter exists
            if parent_param is not None:
                parent_conditions = [
                    EqualsCondition(
                        child=param,
                        parent=parent_param,
                        value=parent_value,
                    )
                    for param in params
                ]
                conditions.extend(parent_conditions)

            cs.add(params + conditions)

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: Dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> "ISAC":
            """
            Create ISAC instance from configuration.

            Args:
                configuration (Configuration): Configuration object.
                cs_transform (Dict[str, dict]): Transform dictionary.
                pre_prefix (str): Prefix for parameter names.
                **kwargs: Additional keyword arguments.

            Returns:
                ISAC: Configured ISAC instance.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{ISAC.PREFIX}"
            else:
                prefix = ISAC.PREFIX

            mode = configuration[f"{prefix}:mode"]
            fallback_strategy = configuration[f"{prefix}:fallback_strategy"]

            # Only get clustering parameters if mode is clustering
            if mode == "clustering":
                n_clusters = configuration[f"{prefix}:n_clusters"]
                distance_threshold = configuration[f"{prefix}:distance_threshold"]
            else:
                n_clusters = 10  # Default values for regression mode
                distance_threshold = 1.0

            return ISAC(
                mode=mode,
                n_clusters=n_clusters,
                distance_threshold=distance_threshold,
                fallback_strategy=fallback_strategy,
                **kwargs,
            )