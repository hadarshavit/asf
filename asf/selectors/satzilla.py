from typing import Any

import numpy as np
import pandas as pd

from asf.epm import EPM
from asf.predictors.random_forest import RandomForestClassifierWrapper
from asf.selectors.abstract_epm_based_selector import AbstractEPMBasedSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class SATzilla(AbstractEPMBasedSelector, AbstractModelBasedSelector):
    """
    SATzilla-like selector using Schmee & Hahn (1979) iterative imputation
    for censored runtimes (log-scale) and per-algorithm ridge models on
    expanded features (original + pairwise products).
    -> Feature Selection is recommended.
    """

    def __init__(
        self,
        model_class: Any = RandomForestClassifierWrapper,
        **kwargs,
    ):
        """
        Initialize the SATzillaSelector.

        Args:
            model_class: Callable returning a fresh model instance.
            model_kwargs: Keyword args forwarded to model.
            use_log10: If True, use base-10 log transform for targets.
            random_state: Random seed for reproducibility.
            em_max_iter: Max iterations for Schmee & Hahn imputation.
            em_tol: Convergence tolerance for imputed target updates.
            em_min_sigma: Minimum residual std to avoid numerical issues.
            **kwargs: Additional args passed to parent.
        """
        super().__init__(model_class=model_class, **kwargs)
        self.epms: dict[str, EPM] = {}

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        labels: pd.DataFrame | pd.Series | list[str] | np.ndarray,
    ) -> None:
        """
        Fit per-algorithm models.

        Args:
            features: DataFrame of instance features (n_instances x n_features).
            performance: DataFrame of runtimes (n_instances x n_algorithms).
            sat_labels: Optional array-like aligned with features.index containing
                        SAT/UNSAT labels; if provided, train per-status models.
        """
        # Normalize labels to a 1D pandas Series aligned with features/performance
        if isinstance(labels, pd.DataFrame):
            if labels.shape[1] != 1:
                raise ValueError("labels DataFrame must have exactly one column")
            labels_series = labels.squeeze(axis=1)
        elif isinstance(labels, pd.Series):
            labels_series = labels
        else:
            # list/ndarray -> Series with same index as features
            labels_series = pd.Series(labels, index=features.index)

        # Ensure index alignment
        if not labels_series.index.equals(features.index):
            labels_series = labels_series.reindex(features.index)

        # Train the label classifier (use underlying sklearn model for proba later)
        self.label_classifier = self.model_class()
        self.label_classifier.fit(features.values, labels_series.values)
        # Get class order consistent with predict_proba outputs
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
                    # No data for this label; skip to avoid training on empty set
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
        if hasattr(self.label_classifier, "model_class") and hasattr(
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
