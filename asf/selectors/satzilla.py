from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from asf.epm import EPM
from asf.predictors.random_forest import RandomForestClassifierWrapper
from asf.predictors.ridge import RidgeRegressorWrapper
from asf.selectors.abstract_epm_based_selector import AbstractEPMBasedSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SATzilla(ConfigurableMixin, AbstractEPMBasedSelector, AbstractModelBasedSelector):
    """
    SATzilla-like selector using iterative imputation for censored runtimes.

    Uses per-algorithm ridge models on expanded features.

    Attributes
    ----------
    epms : dict[str, dict[str, EPM]]
        Mapping from algorithm name to another mapping of label to EPM.
    label_classifier : AbstractPredictor or None
        Model trained to predict instance labels (e.g., SAT/UNSAT).
    labels : list[str]
        Unique labels used for conditioning EPMs.
    """

    PREFIX = "satzilla"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[Any] = RandomForestClassifierWrapper,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SATzilla selector.

        Parameters
        ----------
        model_class : type, default=RandomForestClassifierWrapper
            The class of the model used for label classification.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(model_class=model_class, **kwargs)
        self.epms: dict[str, dict[str, EPM]] = {}
        self.label_classifier: Any = None
        self.labels: list[str] = []

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        labels: pd.DataFrame | pd.Series | list[str] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit per-algorithm models.

        Parameters
        ----------
        features : pd.DataFrame
            Training features (instances x features).
        performance : pd.DataFrame
            Training performance matrix (instances x algorithms).
        labels : pd.DataFrame, pd.Series, list, or np.ndarray, optional
            Optional labels for training conditioned EPMs.
        """
        if labels is None:
            labels_series = pd.Series(["default"] * len(features), index=features.index)
            self.label_classifier = None
            self.labels = ["default"]
        else:
            if isinstance(labels, pd.DataFrame):
                labels_series = labels.squeeze(axis=1)
            elif isinstance(labels, pd.Series):
                labels_series = labels
            else:
                labels_series = pd.Series(labels, index=features.index)

            if not labels_series.index.equals(features.index):
                labels_series = labels_series.reindex(features.index)

            self.label_classifier = self.model_class()
            self.label_classifier.fit(features.values, labels_series.values)

            # Extract unique labels
            if hasattr(self.label_classifier, "model_class") and hasattr(
                self.label_classifier.model_class, "classes_"
            ):
                self.labels = [
                    str(c) for c in self.label_classifier.model_class.classes_
                ]
            else:
                self.labels = [str(c) for c in np.unique(labels_series.values)]

        for algo in self.algorithms:
            self.epms[str(algo)] = {}
            for label in self.labels:
                idx = labels_series.astype(str) == str(label)
                if idx.sum() == 0:
                    continue
                self.epms[str(algo)][str(label)] = EPM(**self.epm_kwargs)
                X_sub = features.loc[idx]
                y_sub = performance.loc[idx, algo]
                self.epms[str(algo)][str(label)].fit(X_sub, y_sub)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance.

        Parameters
        ----------
        features : pd.DataFrame or None
            The input features.
        performance : pd.DataFrame or None, default=None
            Partial performance data.

        Returns
        -------
        dict
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("SATzilla requires features for prediction.")
        n_instances = features.shape[0]
        n_algorithms = len(self.algorithms)
        preds = np.zeros((n_instances, n_algorithms), dtype=float)

        if self.label_classifier is None:
            label_probs = np.ones((n_instances, 1), dtype=float)
        elif hasattr(self.label_classifier, "model_class") and hasattr(
            self.label_classifier.model_class, "predict_proba"
        ):
            label_probs = self.label_classifier.model_class.predict_proba(
                features.values
            )
        else:
            hard_preds = np.asarray(self.label_classifier.predict(features.values))
            classes = np.asarray(self.labels)
            label_probs = (hard_preds[:, None] == classes[None, :]).astype(float)

        for j, algo in enumerate(self.algorithms):
            for k, label in enumerate(self.labels):
                if str(algo) not in self.epms or str(label) not in self.epms[str(algo)]:
                    continue
                pred_time = np.asarray(
                    self.epms[str(algo)][str(label)].predict(features)
                )
                preds[:, j] += label_probs[:, k] * pred_time

        best_idx = np.argmin(preds, axis=1)
        results: dict[str, list[tuple[str, float]]] = {}
        for i, inst in enumerate(features.index):
            j = int(best_idx[i])
            algo = str(self.algorithms[j])
            results[str(inst)] = [(algo, float(self.budget or 0))]
        return results

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
                Define hyperparameters for SATzilla.

                Parameters
                ----------
                model_class : list[type] or None, default=None
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
            model_class = [RidgeRegressorWrapper]

        model_class_param = ClassChoice(
            name="model_class",
            choices=model_class,
            default=model_class[0],
        )

        use_log10_param = Categorical(
            name="use_log10",
            items=[True, False],
            default=True,
        )

        em_max_iter_param = Integer(
            name="em_max_iter",
            bounds=(5, 50),
            default=20,
        )

        em_tol_param = Float(
            name="em_tol",
            bounds=(1e-6, 1e-2),
            log=True,
            default=1e-3,
        )

        em_min_sigma_param = Float(
            name="em_min_sigma",
            bounds=(1e-8, 1e-1),
            log=True,
            default=1e-6,
        )

        params = [
            model_class_param,
            use_log10_param,
            em_max_iter_param,
            em_tol_param,
            em_min_sigma_param,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[SATzilla]:
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
            Partial function for SATzilla.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SATzilla, **config)
