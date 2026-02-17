from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.xgboost import XGBoostRankerWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import ConfigurationSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class DyadRanking(ConfigurableMixin, AbstractModelBasedSelector):
    """
    Dyad Ranking for Algorithm Selection.

    Implements the approach from:
    Tornede et al. (2019) "Algorithm Selection as Recommendation:
    From Collaborative Filtering to Dyad Ranking"

    Uses XGBoost instead of PLNet for ranking, but the overall approach is the same.


    How it works:
    1. Create cross-product: instance_features x algorithm_features
    2. Train ranker to predict which dyads rank higher
    3. For new instance: score all dyads, pick best algorithm
    """

    PREFIX = "dyad_ranking"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = XGBoostRankerWrapper,
        algorithm_features: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize DyadRanking selector.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=XGBoostRankerWrapper
            The ranking model class to use.
        algorithm_features : pd.DataFrame or None, default=None
            Pre-computed algorithm features. If None, uses one-hot encoding.
            Shape: [n_algorithms, n_algo_features]
            Index: algorithm names
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier: AbstractPredictor | None = None
        if algorithm_features is not None:
            self.algorithm_features = algorithm_features

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Fit the dyad ranking model.

        Creates dyads by combining instance features with algorithm features,
        then trains a ranker to predict algorithm rankings per instance.

        Parameters
        ----------
        features : pd.DataFrame
            Instance features. Shape: [n_instances, n_instance_features]
        performance : pd.DataFrame
            Algorithm performance per instance. Shape: [n_instances, n_algorithms]
        """
        # Step 1: Prepare algorithm features
        if self.algorithm_features is None:
            self.algorithm_features = self._create_onehot_algorithm_features()
        else:
            # User-provided algorithm features - ensure they match our algorithm set
            if not all(
                algo in self.algorithm_features.index for algo in self.algorithms
            ):
                raise ValueError(
                    "Provided algorithm_features must include all algorithms in the dataset"
                )
            # Reorder to match self.algorithms and make a copy
            self.algorithm_features = self.algorithm_features.loc[
                list(self.algorithms)
            ].copy()

        # Step 2: Create dyads (cross-product of instances × algorithms)
        dyads_df = self._create_dyads(features, performance)

        # Step 3: Convert performance to ranks per instance
        dyads_df = self._add_rankings(dyads_df)

        # Step 4: Prepare training data
        X, y, qid = self._prepare_training_data(dyads_df)

        # Step 5: Train ranking model
        self.classifier = self.model_class()
        if self.classifier is None:
            raise RuntimeError("Classifier could not be initialized.")

        self.classifier.fit(X, y, qid=qid)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance.

        Parameters
        ----------
        features : pd.DataFrame
            Instance features for prediction.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
            Format: {instance_name: [(algorithm_name, budget)]}
        """
        if features is None:
            raise ValueError("DyadRanking requires features for prediction.")
        if self.classifier is None:
            raise RuntimeError("Classifier has not been fitted.")
        if self.algorithm_features is None:
            raise RuntimeError("Algorithm features missing.")

        # Create dyads for prediction
        dyads_df = self._create_dyads_for_prediction(features)

        # Get features in correct order
        feature_cols = list(self.features) + list(self.algorithm_features.columns)
        X = dyads_df[feature_cols]

        # Predict scores
        predictions = self.classifier.predict(X)
        dyads_df["predicted_score"] = predictions

        # Select best algorithm per instance
        results: dict[str, list[tuple[str, float]]] = {}
        for instance_name in features.index:
            instance_dyads = dyads_df[dyads_df["INSTANCE_ID"] == instance_name]

            # Get best algorithm (lowest predicted rank)
            best_idx = instance_dyads["predicted_score"].argmin()
            best_algo = instance_dyads.iloc[best_idx]["ALGORITHM"]

            results[str(instance_name)] = [(str(best_algo), float(self.budget or 0))]

        return results

    def _create_onehot_algorithm_features(self) -> pd.DataFrame:
        """Create one-hot encoded algorithm features."""
        encoder = OneHotEncoder(sparse_output=False)
        algo_array = np.array(self.algorithms).reshape(-1, 1)
        encoded = encoder.fit_transform(algo_array)

        return pd.DataFrame(
            encoded,
            index=list(self.algorithms),
            columns=[f"algo_{i}" for i in range(len(self.algorithms))],
        )

    def _create_dyads(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create dyads: cross-product of instances and algorithms.

        Each dyad combines:
        - Instance features (problem characteristics)
        - Algorithm features (algorithm characteristics)
        - Performance value (ground truth)
        """
        # Prepare instance features
        features_reset = features[list(self.features)].reset_index()
        inst_col_name = features.index.name or "index"
        features_reset = features_reset.rename(columns={inst_col_name: "INSTANCE_ID"})

        # Prepare algorithm features
        algo_features = self.algorithm_features.copy()
        algo_features.index.name = "ALGORITHM"
        algo_features_reset = algo_features.reset_index()

        # Create cross-product
        dyads = pd.merge(features_reset, algo_features_reset, how="cross")

        # Add performance values
        performance_reset = performance[self.algorithms].reset_index()
        performance_reset = performance_reset.rename(
            columns={performance.index.name or "index": "INSTANCE_ID"}
        )
        performance_stacked = performance_reset.melt(
            id_vars="INSTANCE_ID", var_name="ALGORITHM", value_name="PERFORMANCE"
        )

        # Merge with dyads
        dyads = dyads.merge(performance_stacked, on=["INSTANCE_ID", "ALGORITHM"])

        return dyads

    def _create_dyads_for_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create dyads for prediction (without performance values)."""
        # Prepare instance features
        features_reset = features[list(self.features)].reset_index()
        inst_col_name = features.index.name or "index"
        features_reset = features_reset.rename(columns={inst_col_name: "INSTANCE_ID"})

        # Prepare algorithm features
        algo_features = self.algorithm_features.copy()
        algo_features.index.name = "ALGORITHM"
        algo_features_reset = algo_features.reset_index()

        # Create cross-product
        dyads = pd.merge(features_reset, algo_features_reset, how="cross")

        return dyads

    def _add_rankings(self, dyads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert performance values to rankings per instance.

        Rank 1 = best performing algorithm
        Rank 2 = second best, etc.
        """
        ranked_groups = []

        for instance_name, group in dyads_df.groupby("INSTANCE_ID"):
            group = group.copy()
            # Rank: lower is better (rank 1 = best)
            group["rank"] = group["PERFORMANCE"].rank(
                ascending=not self.maximize, method="min"
            )
            ranked_groups.append(group)

        return pd.concat(ranked_groups, ignore_index=True)

    def _prepare_training_data(
        self,
        dyads_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Prepare training data for ranking model.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (instance features + algorithm features)
        y : pd.Series
            Target ranks
        qid : np.ndarray
            Group IDs (instance IDs) for ranking
        """
        # Features: instance features + algorithm features
        feature_cols = list(self.features) + list(self.algorithm_features.columns)
        X = dyads_df[feature_cols]

        # Target: ranks
        y = dyads_df["rank"]

        # Group by instance (qid)
        q_encoder = OrdinalEncoder()
        qid = (
            q_encoder.fit_transform(dyads_df["INSTANCE_ID"].to_numpy().reshape(-1, 1))
            .flatten()
            .astype(int)
        )

        return X, y, qid

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for DyadRanking.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]] or None, default=None
            List of model classes to choose from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if model_class is None:
            model_class = [XGBoostRankerWrapper]

        model_class_param = ClassChoice(
            name="model_class",
            choices=model_class,
            default=model_class[0],
        )

        return [model_class_param], [], []
