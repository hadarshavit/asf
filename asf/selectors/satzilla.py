from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from asf.epm import EPM
from asf.predictors.random_forest import RandomForestClassifierWrapper
from asf.predictors.ridge import RidgeRegressorWrapper
from asf.selectors.abstract_epm_based_selector import AbstractEPMBasedSelector
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

# Add optional ConfigSpace import (like other selectors)
try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Float,
        EqualsCondition,
        Configuration,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SATzilla(AbstractEPMBasedSelector, AbstractModelBasedSelector):
    """
    SATzilla-like selector using Schmee & Hahn (1979) iterative imputation
    for censored runtimes (log-scale) and per-algorithm ridge models on
    expanded features (original + pairwise products).
    -> Feature Selection is recommended.
    """

    PREFIX = "satzilla"

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

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            cs_transform: Optional[Dict[str, Dict[str, type]]] = None,
            model_class: List[type] = None,
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
            **kwargs,
        ) -> Tuple[ConfigurationSpace, Dict[str, Dict[str, type]]]:
            """
            Build ConfigSpace for SATzilla, including:
            - model_class choice (wrappers) with nested model hyperparams
            - SATzilla-specific EM/log parameters
            """
            if cs is None:
                cs = ConfigurationSpace()
            if cs_transform is None:
                cs_transform = {}
            if model_class is None:
                model_class = [RidgeRegressorWrapper]

            # Prefix for namespacing
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SATzilla.PREFIX}"
            else:
                prefix = SATzilla.PREFIX

            # Model class choice
            model_class_param = Categorical(
                name=f"{prefix}:model_class",
                items=[str(c.__name__) for c in model_class],
            )
            cs_transform[f"{prefix}:model_class"] = {
                str(c.__name__): c for c in model_class
            }

            # SATzilla-specific params
            use_log10 = Categorical(
                f"{prefix}:use_log10",
                [True, False],
                default=True,
            )
            em_max_iter = Integer(
                f"{prefix}:em_max_iter",
                (5, 50),
                default=20,
            )
            em_tol = Float(
                f"{prefix}:em_tol",
                (1e-6, 1e-2),
                log=True,
                default=1e-3,
            )
            em_min_sigma = Float(
                f"{prefix}:em_min_sigma",
                (1e-8, 1e-1),
                log=True,
                default=1e-6,
            )

            params = [model_class_param, use_log10, em_max_iter, em_tol, em_min_sigma]

            # Activate these params only when the parent selector is SATzilla
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

            # Add nested model space under the model_class choice
            for mc in model_class:
                mc.get_configuration_space(
                    cs=cs,
                    pre_prefix=f"{prefix}:model_class",
                    parent_param=model_class_param,
                    parent_value=str(mc.__name__),
                    **kwargs,
                )

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: Dict[str, Dict[str, type]],
            pre_prefix: str = "",
            **kwargs,
        ):
            """
            Instantiate SATzilla from a ConfigSpace configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SATzilla.PREFIX}"
            else:
                prefix = SATzilla.PREFIX

            # Resolve model class wrapper and its constructor from configuration
            model_cls = cs_transform[f"{prefix}:model_class"][
                configuration[f"{prefix}:model_class"]
            ]
            model_ctor = model_cls.get_from_configuration(
                configuration, pre_prefix=f"{prefix}:model_class"
            )

            # Read SATzilla-specific params
            use_log10 = configuration[f"{prefix}:use_log10"]
            em_max_iter = configuration[f"{prefix}:em_max_iter"]
            em_tol = configuration[f"{prefix}:em_tol"]
            em_min_sigma = configuration[f"{prefix}:em_min_sigma"]

            return SATzilla(
                model_class=model_ctor,
                use_log10=use_log10,
                em_max_iter=em_max_iter,
                em_tol=em_tol,
                em_min_sigma=em_min_sigma,
                **kwargs,
            )
