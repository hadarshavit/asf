from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asf.epm import EPM
from asf.predictors.random_forest import RandomForestClassifierWrapper
from asf.predictors.ridge import RidgeRegressorWrapper
from asf.selectors.abstract_epm_based_selector import AbstractEPMBasedSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

# Optional ConfigSpace import (consistent with other modules)
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    from ConfigSpace import (
        Categorical,
        Integer,
        Float,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial


class SATzilla(ConfigurableMixin, AbstractEPMBasedSelector, AbstractModelBasedSelector):
    """
    SATzilla-like selector using Schmee & Hahn (1979) iterative imputation
    for censored runtimes (log-scale) and per-algorithm ridge models on
    expanded features (original + pairwise products).
    -> Feature Selection is recommended.
    """

    PREFIX = "satzilla"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: Any = RandomForestClassifierWrapper,
        **kwargs,
    ):
        """
        Initialize the SATzillaSelector.

        Args:
            model_class: Callable returning a fresh model instance.
            **kwargs: Additional args passed to parent.
        """
        super().__init__(model_class=model_class, **kwargs)
        self.epms: dict[str, EPM] = {}

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        labels: pd.DataFrame | pd.Series | list[str] | np.ndarray | None = None,
    ) -> None:
        """
        Fit per-algorithm models.

        Args:
            features: DataFrame of instance features (n_instances x n_features).
            performance: DataFrame of runtimes (n_instances x n_algorithms).
            labels: Array-like aligned with features.index containing
                    SAT/UNSAT labels; if provided, train per-status models.
                   If None, train a single EPM per algorithm without conditioning.
        """
        if labels is None:
            labels_series = pd.Series(["default"] * len(features), index=features.index)
            self.label_classifier = None
            self.labels = ["default"]
        else:
            # Normalize labels to a 1D pandas Series aligned with features/performance
            if isinstance(labels, pd.DataFrame):
                if labels.shape[1] != 1:
                    raise ValueError("labels DataFrame must have exactly one column")
                labels_series = labels.squeeze(axis=1)
            elif isinstance(labels, pd.Series):
                labels_series = labels
            else:
                labels_series = pd.Series(labels, index=features.index)

            # Ensure index alignment
            if not labels_series.index.equals(features.index):
                labels_series = labels_series.reindex(features.index)

            self.label_classifier = self.model_class()
            self.label_classifier.fit(features.values, labels_series.values)
            if hasattr(self.label_classifier, "model_class") and hasattr(
                self.label_classifier.model_class, "classes_"
            ):
                self.labels = self.label_classifier.model_class.classes_
            else:
                self.labels = np.unique(labels_series.values)

        # Train per-algorithm EPMs conditioned on label
        for algo in self.algorithms:
            self.epms[algo] = {}
            for label in self.labels:
                idx = labels_series == label
                if idx.sum() == 0:
                    continue
                self.epms[algo][label] = EPM(**self.epm_kwargs)
                X_sub = features.loc[idx]
                y_sub = performance.loc[idx, algo]
                self.epms[algo][label].fit(X_sub, y_sub)

    def _predict(
        self,
        features: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str | None, float]]]:
        """
        Predict best algorithm per instance.

        Args:
            features: DataFrame of instance features to predict for.

        Returns:
            Mapping instance_name -> [(algorithm_name_or_None, budget)].
        """

        n_instances = features.shape[0]
        n_algorithms = len(self.algorithms)
        preds = np.zeros((n_instances, n_algorithms), dtype=float)

        # Get label probabilities once; use underlying sklearn model
        if self.label_classifier is None:
            # No label classifier: uniform probability for the single "default" label
            label_probs = np.ones((n_instances, 1), dtype=float)
        elif hasattr(self.label_classifier, "model_class") and hasattr(
            self.label_classifier.model_class, "predict_proba"
        ):
            label_probs = self.label_classifier.model_class.predict_proba(
                features.values
            )
        else:
            # Fallback: use hard predictions and one-hot encode
            hard_preds = np.asarray(self.label_classifier.predict(features.values))
            classes = np.asarray(self.labels)
            label_probs = (hard_preds[:, None] == classes[None, :]).astype(float)

        for j, algo in enumerate(self.algorithms):
            for k, label in enumerate(self.labels):
                # Skip labels with no trained EPM for this algo
                if algo not in self.epms or label not in self.epms[algo]:
                    continue
                pred_time = np.asarray(self.epms[algo][label].predict(features))
                preds[:, j] += label_probs[:, k] * pred_time

        best_idx = np.argmin(preds, axis=1)
        results = {}
        for i, inst in enumerate(features.index):
            j = int(best_idx[i])
            algo = self.algorithms[j]
            results[inst] = [(algo, self.budget)]
        return results

    @staticmethod
    def _define_hyperparameters(model_class=None, **kwargs):
        """Define hyperparameters for SATzilla."""
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
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SATzilla, **config)
