from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.xgboost import XGBoostRankerWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

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

    Reference link:
    https://ris.uni-paderborn.de/download/15011/17060/ci_workshop_tornede.pdf

    Uses XGBoost instead of PLNet for ranking, but the overall approach is the same.


    How it works:
    1. Create cross-product: instance_features x algorithm_features
    2. Sample pairwise comparisons per instance (not full rankings)
    3. Train ranker to predict which dyads rank higher
    4. For new instance: score all dyads, pick best algorithm
    """

    PREFIX = "dyad_ranking"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = XGBoostRankerWrapper,
        algorithm_features: pd.DataFrame | None = None,
        n_pairs_per_instance: int = 10,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize DyadRanking selector.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=XGBoostRankerWrapper
            The ranking model class to use.
        algorithm_features : pd.DataFrame or None, default=None
            Algorithm features representing parameters and structural components.
            If None, falls back to one-hot encoding (not recommended per paper).
            Shape: [n_algorithms, n_algo_features]
            Index: algorithm names
        n_pairs_per_instance : int, default=10
            Number of pairwise comparisons to sample per training instance.
        random_state : int, default=42
            Random seed for pairwise sampling reproducibility.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier: AbstractPredictor | None = None
        self.n_pairs_per_instance = n_pairs_per_instance
        self.random_state = int(random_state)
        # Store user-provided algorithm features separately to avoid being overwritten
        # by parent class fit() method
        self._init_algorithm_features = algorithm_features

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
        if (
            self.algorithm_features is None
            and self._init_algorithm_features is not None
        ):
            self.algorithm_features = self._init_algorithm_features

        if self.algorithm_features is None:
            import warnings

            warnings.warn(
                "Using one-hot algorithm features (fallback). Per Tornede et al. (2019), "
                "algorithm features should represent parameters and structural components. "
                "Consider providing meaningful algorithm_features for better performance.",
                UserWarning,
                stacklevel=2,
            )
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

        dyads_df = self._create_pairwise_comparisons(
            features, performance, self.n_pairs_per_instance
        )

        X, y, qid = self._prepare_training_data(dyads_df)

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

        feature_cols = list(self.features) + list(self.algorithm_features.columns)
        X = dyads_df[feature_cols]

        predictions = self.classifier.predict(X)
        dyads_df["predicted_score"] = predictions

        results: dict[str, list[tuple[str, float]]] = {}
        for instance_name in features.index:
            instance_dyads = dyads_df[dyads_df["INSTANCE_ID"] == instance_name]

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
            index=pd.Index(list(self.algorithms)),
            columns=pd.Index([f"algo_{i}" for i in range(len(self.algorithms))]),
        )

    def _create_dyads_for_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create dyads for prediction (without performance values)."""
        features_reset = features[list(self.features)].reset_index()
        inst_col_name = features.index.name or "index"
        features_reset = features_reset.rename(columns={inst_col_name: "INSTANCE_ID"})

        assert self.algorithm_features is not None
        algo_features = self.algorithm_features.copy()
        algo_features.index.name = "ALGORITHM"
        algo_features_reset = algo_features.reset_index()

        dyads = pd.merge(features_reset, algo_features_reset, how="cross")
        return dyads

    def _create_pairwise_comparisons(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        n_pairs: int,
    ) -> pd.DataFrame:
        """
        Sample pairwise comparisons per instance.

        For each instance, randomly samples n_pairs algorithm pairs where
        the two algorithms have different performance.

        Parameters
        ----------
        features : pd.DataFrame
            Instance features
        performance : pd.DataFrame
            Algorithm performance per instance
        n_pairs : int
            Number of pairs to sample per instance

        Returns
        -------
        pd.DataFrame
            Dyads with pairwise rankings (rank 1 = better, rank 2 = worse)
        """
        rng = np.random.RandomState(self.random_state)
        all_pairs = []

        for instance_idx in features.index:
            instance_perf = performance.loc[instance_idx, self.algorithms]
            instance_feats = features.loc[instance_idx, self.features]

            valid_pairs = []
            for i, algo1 in enumerate(self.algorithms):
                for algo2 in self.algorithms[i + 1 :]:
                    perf1 = instance_perf[algo1]
                    perf2 = instance_perf[algo2]

                    if pd.isna(perf1) or pd.isna(perf2) or np.isclose(perf1, perf2):
                        continue

                    if self.maximize:
                        better, worse = (
                            (algo1, algo2) if perf1 > perf2 else (algo2, algo1)
                        )
                    else:
                        better, worse = (
                            (algo1, algo2) if perf1 < perf2 else (algo2, algo1)
                        )

                    valid_pairs.append((better, worse))

            if len(valid_pairs) == 0:
                continue

            n_sample = min(n_pairs, len(valid_pairs))
            sampled_pairs = [
                valid_pairs[i]
                for i in rng.choice(len(valid_pairs), n_sample, replace=False)
            ]

            # Create dyads for sampled pairs
            for better_algo, worse_algo in sampled_pairs:
                # Create two dyads: one ranked 1 (better), one ranked 2 (worse)
                assert self.algorithm_features is not None
                for algo, rank in [(better_algo, 1), (worse_algo, 2)]:
                    dyad_features = dict(instance_feats)
                    dyad_features.update(self.algorithm_features.loc[algo].to_dict())
                    dyad_features["INSTANCE_ID"] = instance_idx
                    dyad_features["ALGORITHM"] = algo
                    dyad_features["rank"] = rank
                    dyad_features["pair_id"] = (
                        f"{instance_idx}_{better_algo}_{worse_algo}"
                    )
                    all_pairs.append(dyad_features)

        if len(all_pairs) == 0:
            raise ValueError(
                "No valid pairwise comparisons could be created. "
                "This typically happens when all algorithms have identical performance "
                "(e.g., all timeout on all instances). Cannot train dyad ranking model."
            )

        return pd.DataFrame(all_pairs)

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
            Group IDs for ranking (pair IDs for pairwise comparisons)
        """
        # Sort by pair_id to ensure qid is in non-decreasing order (required by XGBoost)
        dyads_df = dyads_df.sort_values("pair_id").reset_index(drop=True)

        assert self.algorithm_features is not None
        feature_cols = list(self.features) + list(self.algorithm_features.columns)
        X = dyads_df[feature_cols]
        y = dyads_df["rank"]

        # Group by pair ID (each pairwise comparison is a separate ranking)
        q_encoder = OrdinalEncoder()
        qid = (
            q_encoder.fit_transform(dyads_df["pair_id"].to_numpy().reshape(-1, 1))
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
        Define hyperparameters for SimpleRanking.

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
            choices=cast(list[type | bool], model_class),
            default=model_class[0],
        )

        return [model_class_param], [], []
