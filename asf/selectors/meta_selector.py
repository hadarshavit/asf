from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class MetaSelector(ConfigurableMixin, AbstractSelector):
    """
    Meta-selector that ensembles multiple base selectors.

    Trains multiple base selectors and uses another selector (the meta-selector)
    to choose among them for each instance based on their out-of-fold performance.

    References
    ----------
    Feurer, M., et al. (2021).
    "Meta-Algorithm Selection."
    https://arxiv.org/abs/2107.09414

    Important: All base selectors and the meta selector must have RETURN_TYPE 'single'
    """

    base_selectors: list[AbstractSelector]
    meta_selector: AbstractSelector
    _selector_map: dict[str, AbstractSelector]
    base_selectors_: list[AbstractSelector]

    PREFIX = "meta"
    RETURN_TYPE = "single"

    @staticmethod
    def _clone_selector(selector: AbstractSelector) -> AbstractSelector:
        """
        Create a fresh selector copy while preserving constructor configuration.

        Deep copy is preferred here because several selectors do not expose a
        reliable parameter introspection API, and ``sel.__class__()`` would
        silently reset user-provided hyperparameters to defaults.
        """
        return copy.deepcopy(selector)

    def __init__(
        self,
        base_selectors: Sequence[AbstractSelector]
        | Callable[[], list[AbstractSelector]]
        | None = None,
        meta_selector: AbstractSelector | Callable[[], AbstractSelector] | None = None,
        candidate_selectors: list[type] | None = None,
        par_factor: int = 10,
        n_folds: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MetaSelector.

        Parameters
        ----------
        base_selectors : Sequence[AbstractSelector] or Callable or None, default=None
            List of base selectors to ensemble, a callable that returns a list of
            selectors, or None if candidate_selectors is provided.
        meta_selector : AbstractSelector or Callable or None, default=None
            The selector to use for choosing among base selectors, or a callable
            that returns a selector instance.
        candidate_selectors : list[type] or None, default=None
            List of selector classes to instantiate as base selectors if
            base_selectors is None.
        par_factor : int, default=10
            Penalty factor for timeout instances when computing meta-features.
            Penalty = budget * par_factor.
        n_folds : int, default=5
            Number of folds for cross-validation during meta-selector training.
        random_state : int, default=42
            Random seed for the KFold cross-validation splitter.
        **kwargs : Any
            Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        if base_selectors is None and candidate_selectors is not None:
            base_selectors = []
            for sel_cls in candidate_selectors:
                try:
                    base_selectors.append(sel_cls())
                except Exception:
                    pass

        if callable(base_selectors):
            self.base_selectors = cast(Any, base_selectors)()
        else:
            self.base_selectors = (
                list(base_selectors) if base_selectors is not None else []
            )

        if callable(meta_selector):
            self.meta_selector = cast(Any, meta_selector)()
        else:
            self.meta_selector = cast(AbstractSelector, meta_selector)

        if not self.base_selectors:
            raise ValueError("`base_selectors` list cannot be empty.")
        if self.meta_selector is None:
            raise ValueError("`meta_selector` cannot be None.")

        for sel in self.base_selectors:
            if getattr(sel, "RETURN_TYPE", None) != "single":
                raise ValueError(
                    f"Base selector {sel.__class__.__name__} must have RETURN_TYPE 'single'."
                )
        if getattr(self.meta_selector, "RETURN_TYPE", None) != "single":
            raise ValueError("Meta selector must have RETURN_TYPE 'single'.")

        self.par_factor = int(par_factor)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)

        self.selector_names = [
            f"{s.__class__.__name__}_{i}" for i, s in enumerate(self.base_selectors)
        ]
        self._selector_map: dict[str, AbstractSelector] = {
            name: sel for name, sel in zip(self.selector_names, self.base_selectors)
        }
        self.base_selectors_: list[AbstractSelector] | None = None

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the MetaSelector.

        Parameters
        ----------
        features : pd.DataFrame
            Training features (instances x features).
        performance : pd.DataFrame
            Training performance (instances x algorithms).
        """
        penalty = float(self.budget or 0) * float(self.par_factor)
        n_instances = len(features)
        meta_performance = pd.DataFrame(
            index=features.index,
            columns=pd.Index(list(self.selector_names)),
            dtype=float,
        )
        meta_performance[:] = np.nan

        n_splits = min(self.n_folds, max(2, n_instances))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(np.arange(n_instances)):
            train_ix = features.index[train_idx]
            val_ix = features.index[val_idx]

            X_train = features.loc[train_ix]
            Y_train = performance.loc[train_ix]
            X_val = features.loc[val_ix]

            for sel_idx, sel in enumerate(self.base_selectors):
                sel_copy = self._clone_selector(sel)
                sel_copy.fit(X_train, Y_train)
                preds = cast(dict[str, Any], sel_copy.predict(X_val))

                col = self.selector_names[sel_idx]
                for inst in val_ix:
                    pred = preds.get(str(inst), [])
                    if not pred:
                        meta_performance.at[inst, col] = penalty
                        continue
                    algo_name, _ = pred[0]
                    rt = performance.at[inst, algo_name]
                    if pd.isna(rt) or float(rt) >= float(self.budget or 0):
                        meta_performance.at[inst, col] = penalty
                    else:
                        meta_performance.at[inst, col] = float(rt)

        meta_performance.fillna(penalty, inplace=True)

        self.base_selectors_ = []
        for sel in self.base_selectors:
            sel_full = self._clone_selector(sel)
            sel_full.fit(features, performance)
            self.base_selectors_.append(sel_full)

        self._selector_map = {
            name: sel for name, sel in zip(self.selector_names, self.base_selectors_)
        }

        self.meta_selector.fit(features, meta_performance)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict with the fitted MetaSelector.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for the test instances.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        meta_predictions = cast(dict[str, Any], self.meta_selector.predict(features))

        if features is None:
            raise ValueError("MetaSelector require features for prediction.")
        final_predictions: dict[str, list[tuple[str, float]]] = {}
        for instance_name, chosen_selector_list in meta_predictions.items():
            if not chosen_selector_list:
                final_predictions[str(instance_name)] = []
                continue

            chosen_selector_name = str(chosen_selector_list[0][0])
            chosen_selector = self._selector_map.get(chosen_selector_name)
            if chosen_selector is None:
                final_predictions[str(instance_name)] = []
                continue

            # Convert instance_name back to original index type if necessary
            if features.index.dtype != object:
                orig_instance_name = features.index.dtype.type(instance_name)
            else:
                orig_instance_name = instance_name
            instance_features = features.loc[[orig_instance_name]]
            final_prediction = cast(
                dict[str, Any], chosen_selector.predict(instance_features)
            )
            preds = final_prediction.get(str(instance_name), [])

            formatted = []
            for entry in preds:
                algo, _ = entry
                formatted.append((str(algo), float(self.budget or 0)))

            final_predictions[str(instance_name)] = formatted
        return final_predictions

    @staticmethod
    def _define_hyperparameters(
        candidate_selectors: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for MetaSelector.

        Parameters
        ----------
        candidate_selectors : list[type] or None, default=None
            List of selector classes to choose from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if candidate_selectors is None:
            return [], [], []

        meta_selector_param = ClassChoice(
            name="meta_selector",
            choices=cast(list[type | bool], candidate_selectors),
            default=candidate_selectors[0],
        )

        par_factor_param = Integer(
            name="par_factor",
            bounds=(1, 100),
            default=10,
        )

        n_folds_param = Integer(
            name="n_folds",
            bounds=(2, 10),
            default=5,
        )

        params = [
            meta_selector_param,
            par_factor_param,
            n_folds_param,
        ]

        return params, [], []
