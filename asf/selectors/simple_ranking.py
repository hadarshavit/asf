from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.xgboost import XGBoostRankerWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SimpleRanking(ConfigurableMixin, AbstractModelBasedSelector):
    """
    Algorithm Selection via Ranking.

    Attributes
    ----------
    classifier : AbstractPredictor or None
        The trained ranking model.
    """

    PREFIX = "simple_ranking"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = XGBoostRankerWrapper,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SimpleRanking.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=XGBoostRankerWrapper
            The class of the ranking model to be used.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier: AbstractPredictor | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Fit the ranking model.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        """
        if self.algorithm_features is None:
            encoder = OneHotEncoder(sparse_output=False)
            self.algorithm_features = pd.DataFrame(
                encoder.fit_transform(np.array(self.algorithms).reshape(-1, 1)),
                index=list(self.algorithms),  # type: ignore[arg-type]
                columns=[f"algo_{i}" for i in range(len(self.algorithms))],  # type: ignore[arg-type]
            )

        performance = performance[self.algorithms]
        features = features[list(self.features)]

        # Reset index to have instance names as a column for merging
        features_reset = features.reset_index().rename(
            columns={features.index.name or "index": "INSTANCE_ID"}
        )
        self.algorithm_features.index.name = "ALGORITHM"

        # Create cross-product of instances and algorithms
        total_features = pd.merge(
            features_reset, self.algorithm_features.reset_index(), how="cross"
        )

        stacked_perf = performance.stack().reset_index()
        stacked_perf.columns = ["INSTANCE_ID", "ALGORITHM", "PERFORMANCE"]

        merged = total_features.merge(
            stacked_perf, on=["INSTANCE_ID", "ALGORITHM"], how="left"
        )

        # Calculate ranks per instance
        gdfs = []
        for _name, gdf in merged.groupby("INSTANCE_ID"):
            gdf["rank"] = gdf["PERFORMANCE"].rank(
                ascending=not self.maximize, method="min"
            )
            gdfs.append(gdf)
        merged = pd.concat(gdfs)

        # Features for training
        X = merged.drop(columns=["INSTANCE_ID", "ALGORITHM", "PERFORMANCE", "rank"])
        y = merged["rank"]

        q_encoder = OrdinalEncoder()
        qid = q_encoder.fit_transform(
            merged["INSTANCE_ID"].to_numpy().reshape(-1, 1)
        ).flatten()

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
            The input features.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if features is None:
            raise ValueError("SimpleRanking require features for prediction.")
        if self.classifier is None:
            raise RuntimeError("Classifier has not been fitted.")
        if self.algorithm_features is None:
            raise RuntimeError("Algorithm features missing.")

        f_cols = list(self.features)
        inst_name = features.index.name or "index"
        features_reset = features.reset_index().rename(
            columns={inst_name: "INSTANCE_ID"}
        )

        total_features = pd.merge(
            features_reset, self.algorithm_features.reset_index(), how="cross"
        )

        X = total_features[f_cols + list(self.algorithm_features.columns)]
        predictions = self.classifier.predict(X)
        results: dict[str, list[tuple[str, float]]] = {}
        for i, instance_name in enumerate(features.index):
            mask = total_features["INSTANCE_ID"] == instance_name
            # Local best for this group
            best_idx = int(np.argmin(predictions[mask]))
            best_algo = total_features[mask].iloc[best_idx]["ALGORITHM"]
            results[str(instance_name)] = [(str(best_algo), float(self.budget or 0))]

        return results

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
            choices=model_class,
            default=model_class[0],
        )

        return [model_class_param], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[SimpleRanking]:
        """
        Create a partial function from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            The clean configuration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for SimpleRanking.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SimpleRanking, **config)
