"""Ensemble selectors for algorithm selection.

This module provides ensemble methods for combining multiple algorithm selectors:
- BaggingSelector: Bootstrap aggregating of a single selector type
- VotingSelector: Hard voting across heterogeneous selectors
- StackingSelector: Meta-learning using base selector outputs as features
"""

from __future__ import annotations

import copy
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        Integer,
        UniformFloatHyperparameter,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


def _has_generate_features(selector: AbstractSelector) -> bool:
    """Check if a selector can generate features.

    Parameters
    ----------
    selector : AbstractSelector
        The selector to check.

    Returns
    -------
    bool
        True if the selector has a generate_features method.
    """
    return isinstance(selector, AbstractFeatureGenerator) and callable(
        getattr(selector, "generate_features", None)
    )


def _clone_selector(selector: AbstractSelector) -> AbstractSelector:
    """Create a fresh copy of a selector.

    Parameters
    ----------
    selector : AbstractSelector
        The selector to clone.

    Returns
    -------
    AbstractSelector
        A new instance of the selector.
    """
    try:
        return selector.__class__()
    except Exception:
        return copy.deepcopy(selector)


class BaggingSelector(ConfigurableMixin, AbstractSelector):
    """Bagging ensemble for algorithm selectors.

    Trains multiple instances of the same base selector on bootstrap
    samples of the training data and aggregates predictions via voting.

    Attributes
    ----------
    base_selector : AbstractSelector
        The base selector to bag.
    n_estimators : int
        Number of bootstrap samples/selectors to train.
    sample_fraction : float
        Fraction of samples to use in each bootstrap sample.
    random_state : int or None
        Random state for reproducibility.
    estimators_ : list[AbstractSelector]
        Fitted estimators after calling fit().
    """

    PREFIX = "bagging"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selector: AbstractSelector | Callable[[], AbstractSelector] | None = None,
        n_estimators: int = 10,
        sample_fraction: float = 1.0,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaggingSelector.

        Parameters
        ----------
        base_selector : AbstractSelector or Callable or None
            The base selector to bag. Can be an instance or a callable
            that returns one.
        n_estimators : int, default=10
            Number of bootstrap samples/selectors to train.
        sample_fraction : float, default=1.0
            Fraction of samples to use in each bootstrap sample.
        random_state : int or None, default=None
            Random state for reproducibility.
        **kwargs : Any
            Additional keyword arguments passed to AbstractSelector.
        """
        super().__init__(**kwargs)

        if callable(base_selector) and not isinstance(base_selector, AbstractSelector):
            self.base_selector = base_selector()
        else:
            self.base_selector = base_selector

        if self.base_selector is None:
            raise ValueError("base_selector cannot be None")

        self.n_estimators = int(n_estimators)
        self.sample_fraction = float(sample_fraction)
        self.random_state = random_state
        self.estimators_: list[AbstractSelector] = []

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Fit the bagging ensemble.

        Parameters
        ----------
        features : pd.DataFrame
            Training features.
        performance : pd.DataFrame
            Algorithm performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = len(features)
        sample_size = max(1, int(n_samples * self.sample_fraction))

        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample with replacement
            indices = rng.choice(n_samples, size=sample_size, replace=True)

            # Use iloc to sample and copy to ensure independent DataFrames
            X_sample = features.iloc[indices].copy()
            X_sample.index = pd.RangeIndex(len(X_sample))
            y_sample = performance.iloc[indices].copy()
            y_sample.index = pd.RangeIndex(len(y_sample))

            assert self.base_selector is not None
            estimator = _clone_selector(self.base_selector)
            estimator.fit(X_sample, y_sample, **kwargs)
            self.estimators_.append(estimator)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Predict using the bagging ensemble.

        Parameters
        ----------
        features : pd.DataFrame or None
            Input features for prediction.
        performance : pd.DataFrame or None, default=None
            Performance data (unused).

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if features is None:
            raise ValueError("BaggingSelector requires features for prediction.")

        if not self.estimators_:
            raise RuntimeError("BaggingSelector has not been fitted.")

        # Collect votes from all estimators
        all_votes: dict[str, dict[str, int]] = {str(idx): {} for idx in features.index}

        for estimator in self.estimators_:
            preds = estimator.predict(features)
            if isinstance(preds, dict):
                for instance, schedule in preds.items():
                    if schedule:
                        assert isinstance(schedule, list)
                        first_item = schedule[0]
                        assert isinstance(first_item, tuple) and len(first_item) >= 1
                        algo = str(first_item[0])
                        votes_for_instance = all_votes[str(instance)]
                        votes_for_instance[algo] = votes_for_instance.get(algo, 0) + 1

        # Aggregate votes
        result: dict[str, list[tuple[str, float]]] = {}
        for instance in features.index:
            votes = all_votes[str(instance)]
            if votes:
                best_algo = max(votes.keys(), key=lambda a: votes[a])
                result[str(instance)] = [(best_algo, float(self.budget or 0))]
            else:
                result[str(instance)] = []

        return result

    @staticmethod
    def _define_hyperparameters(
        base_selector_class: list[type] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for BaggingSelector.

        Parameters
        ----------
        base_selector_class : list[type] or None, default=None
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

        hyperparameters: list[Any] = [
            Integer("n_estimators", bounds=(2, 50), default=10),
            UniformFloatHyperparameter(
                "sample_fraction", lower=0.5, upper=1.0, default_value=1.0
            ),
        ]

        if base_selector_class:
            hyperparameters.append(
                ClassChoice(
                    "base_selector",
                    choices=cast(list[type | bool], base_selector_class),
                )
            )

        return hyperparameters, [], []


class VotingSelector(ConfigurableMixin, AbstractSelector):
    """Voting ensemble for algorithm selectors.

    Combines predictions from multiple base selectors using hard voting
    (majority vote).

    Attributes
    ----------
    selectors : list[AbstractSelector]
        List of base selectors.
    weights : list[float] or None
        Weights for each selector's vote.
    selectors_ : list[AbstractSelector]
        Fitted selectors after calling fit().
    """

    PREFIX = "voting"
    RETURN_TYPE = "single"

    def __init__(
        self,
        selectors: (
            list[AbstractSelector] | Callable[[], list[AbstractSelector]] | None
        ) = None,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the VotingSelector.

        Parameters
        ----------
        selectors : list[AbstractSelector] or Callable or None
            List of base selectors. Can be instances or a callable
            that returns a list.
        weights : list[float] or None, default=None
            Weights for each selector's vote. If None, equal weights.
        **kwargs : Any
            Additional keyword arguments passed to AbstractSelector.
        """
        super().__init__(**kwargs)

        if callable(selectors) and not isinstance(selectors, list):
            self.selectors = selectors()
        else:
            self.selectors = list(selectors) if selectors is not None else []

        if not self.selectors:
            raise ValueError("selectors list cannot be empty")

        self.weights = weights
        if self.weights is not None and len(self.weights) != len(self.selectors):
            raise ValueError("weights must have same length as selectors")

        self.selectors_: list[AbstractSelector] = []

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Fit the voting ensemble.

        Parameters
        ----------
        features : pd.DataFrame
            Training features.
        performance : pd.DataFrame
            Algorithm performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.selectors_ = []

        for selector in self.selectors:
            fitted = _clone_selector(selector)
            fitted.fit(features, performance, **kwargs)
            self.selectors_.append(fitted)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Predict using hard voting.

        Parameters
        ----------
        features : pd.DataFrame or None
            Input features for prediction.
        performance : pd.DataFrame or None, default=None
            Performance data (unused).

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if features is None:
            raise ValueError("VotingSelector requires features for prediction.")

        if not self.selectors_:
            raise RuntimeError("VotingSelector has not been fitted.")

        weights = self.weights or [1.0] * len(self.selectors_)

        # Collect weighted votes
        all_votes: dict[str, dict[str, float]] = {
            str(idx): {} for idx in features.index
        }

        for selector, weight in zip(self.selectors_, weights):
            preds = selector.predict(features)
            if isinstance(preds, dict):
                for instance, schedule in preds.items():
                    if schedule:
                        algo = str(schedule[0][0])
                        all_votes[str(instance)][algo] = (
                            all_votes[str(instance)].get(algo, 0.0) + weight
                        )

        # Select winner by weighted votes
        result: dict[str, list[tuple[str, float]]] = {}
        for instance in features.index:
            votes = all_votes[str(instance)]
            if votes:
                best_algo = max(votes.keys(), key=lambda a: votes[a])
                result[str(instance)] = [(best_algo, float(self.budget or 0))]
            else:
                result[str(instance)] = []

        return result

    @staticmethod
    def _define_hyperparameters(
        selector_classes: list[type] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for VotingSelector.

        Parameters
        ----------
        selector_classes : list[type] or None, default=None
            List of selector classes to include.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        # VotingSelector takes a list of selectors, not configurable via ConfigSpace
        return [], [], []


class StackingSelector(ConfigurableMixin, AbstractSelector):
    """Stacking ensemble for algorithm selectors.

    Trains base selectors and uses their outputs as features for a meta-selector.
    For selectors with generate_features(), can optionally use the generated
    features instead of final predictions.

    Attributes
    ----------
    base_selectors : list[AbstractSelector]
        List of base selectors.
    meta_selector : AbstractSelector
        Meta-selector trained on base selector outputs.
    use_generated_features : bool
        If True, use generate_features() output for selectors that support it.
    cv : int
        Number of cross-validation folds for generating meta-features.
    use_original_features : bool
        If True, include original features in meta-features.
    random_state : int or None
        Random state for cross-validation.
    base_selectors_ : list[AbstractSelector]
        Fitted base selectors.
    meta_selector_ : AbstractSelector
        Fitted meta-selector.
    """

    PREFIX = "stacking"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selectors: (
            list[AbstractSelector] | Callable[[], list[AbstractSelector]] | None
        ) = None,
        meta_selector: AbstractSelector | Callable[[], AbstractSelector] | None = None,
        use_generated_features: bool = False,
        cv: int = 5,
        use_original_features: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the StackingSelector.

        Parameters
        ----------
        base_selectors : list[AbstractSelector] or Callable or None
            List of base selectors.
        meta_selector : AbstractSelector or Callable or None
            Meta-selector for final predictions.
        use_generated_features : bool, default=False
            If True, use generate_features() output for selectors that have it.
            If False, use one-hot encoded final predictions.
        cv : int, default=5
            Number of cross-validation folds.
        use_original_features : bool, default=True
            If True, concatenate original features with stacked features.
        random_state : int or None, default=None
            Random state for cross-validation.
        **kwargs : Any
            Additional keyword arguments passed to AbstractSelector.
        """
        super().__init__(**kwargs)

        if callable(base_selectors) and not isinstance(base_selectors, list):
            self.base_selectors = base_selectors()
        else:
            self.base_selectors = (
                list(base_selectors) if base_selectors is not None else []
            )

        if callable(meta_selector) and not isinstance(meta_selector, AbstractSelector):
            self.meta_selector = meta_selector()
        else:
            self.meta_selector = meta_selector

        if not self.base_selectors:
            raise ValueError("base_selectors list cannot be empty")
        if self.meta_selector is None:
            raise ValueError("meta_selector cannot be None")

        self.use_generated_features = bool(use_generated_features)
        self.cv = int(cv)
        self.use_original_features = bool(use_original_features)
        self.random_state = random_state

        self.base_selectors_: list[AbstractSelector] = []
        self.meta_selector_: AbstractSelector | None = None
        self._encoder: OneHotEncoder | None = None

    def _get_selector_features(
        self,
        selector: AbstractSelector,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get features from a selector for stacking.

        Parameters
        ----------
        selector : AbstractSelector
            The fitted selector.
        features : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Features generated by the selector.
        """
        if self.use_generated_features and _has_generate_features(selector):
            # Use generate_features output
            gen_features = selector.generate_features(features)  # type: ignore[attr-defined]
            if isinstance(gen_features, pd.DataFrame):
                return gen_features
            return pd.DataFrame(gen_features, index=features.index)

        # Fall back to one-hot encoded predictions
        preds = selector.predict(features)
        if isinstance(preds, dict):
            labels = []
            for idx in features.index:
                idx_str = str(idx)
                schedule = preds.get(idx_str) if idx_str in preds else None
                if schedule and isinstance(schedule, list) and len(schedule) > 0:
                    first_item = schedule[0]
                    if isinstance(first_item, tuple) and len(first_item) >= 1:
                        labels.append(str(first_item[0]))
                    else:
                        labels.append(
                            self.algorithms[0] if self.algorithms else "unknown"
                        )
                else:
                    labels.append(self.algorithms[0] if self.algorithms else "unknown")
        else:
            labels = [self.algorithms[0]] * len(features)

        # One-hot encode
        if self._encoder is None:
            self._encoder = OneHotEncoder(
                sparse_output=False,
                categories=[self.algorithms],
                handle_unknown="ignore",
            )
            self._encoder.fit(np.array(self.algorithms).reshape(-1, 1))

        encoded = self._encoder.transform(np.array(labels).reshape(-1, 1))
        return pd.DataFrame(encoded, index=features.index)

    def _build_meta_features(
        self,
        selectors: list[AbstractSelector],
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build meta-features from all base selectors.

        Parameters
        ----------
        selectors : list[AbstractSelector]
            List of fitted selectors.
        features : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Concatenated meta-features from all selectors.
        """
        meta_features_list = []

        for i, selector in enumerate(selectors):
            sel_features = self._get_selector_features(selector, features)
            # Prefix columns to avoid collisions
            sel_features.columns = pd.Index(
                [f"sel{i}_{col}" for col in sel_features.columns]
            )
            meta_features_list.append(sel_features)

        meta_features = pd.concat(meta_features_list, axis=1)

        if self.use_original_features:
            meta_features = pd.concat(
                [features.reset_index(drop=True), meta_features.reset_index(drop=True)],
                axis=1,
            )
            meta_features.index = features.index

        return meta_features

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Fit the stacking ensemble.

        Parameters
        ----------
        features : pd.DataFrame
            Training features.
        performance : pd.DataFrame
            Algorithm performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        n_samples = len(features)
        n_splits = min(self.cv, n_samples)

        # Initialize encoder with all algorithms
        self._encoder = OneHotEncoder(
            sparse_output=False, categories=[self.algorithms], handle_unknown="ignore"
        )
        self._encoder.fit(np.array(self.algorithms).reshape(-1, 1))

        # Cross-validation to generate meta-features without data leakage
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Store out-of-fold predictions for each selector
        oof_features: list[pd.DataFrame] = [
            pd.DataFrame(index=features.index) for _ in self.base_selectors
        ]

        for train_idx, val_idx in kf.split(np.arange(n_samples)):
            train_ix = features.index[train_idx]
            val_ix = features.index[val_idx]

            X_train = features.loc[train_ix]
            y_train = performance.loc[train_ix]
            X_val = features.loc[val_ix]

            for i, selector in enumerate(self.base_selectors):
                fold_selector = _clone_selector(selector)
                fold_selector.fit(X_train, y_train, **kwargs)

                val_features = self._get_selector_features(fold_selector, X_val)
                val_features.columns = pd.Index(
                    [f"sel{i}_{col}" for col in val_features.columns]
                )

                for col in val_features.columns:
                    if col not in oof_features[i].columns:
                        oof_features[i][col] = np.nan
                    oof_features[i].loc[val_ix, col] = val_features[col].values

        # Combine out-of-fold features
        meta_train_features = pd.concat(oof_features, axis=1)
        meta_train_features = meta_train_features.fillna(0)

        if self.use_original_features:
            meta_train_features = pd.concat(
                [
                    features.reset_index(drop=True),
                    meta_train_features.reset_index(drop=True),
                ],
                axis=1,
            )
            meta_train_features.index = features.index

        # Fit base selectors on full data
        self.base_selectors_ = []
        for selector in self.base_selectors:
            fitted = _clone_selector(selector)
            fitted.fit(features, performance, **kwargs)
            self.base_selectors_.append(fitted)

        # Fit meta-selector
        assert self.meta_selector is not None
        self.meta_selector_ = _clone_selector(self.meta_selector)
        self.meta_selector_.fit(meta_train_features, performance, **kwargs)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Predict using the stacking ensemble.

        Parameters
        ----------
        features : pd.DataFrame or None
            Input features for prediction.
        performance : pd.DataFrame or None, default=None
            Performance data (unused).

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if features is None:
            raise ValueError("StackingSelector requires features for prediction.")

        if not self.base_selectors_ or self.meta_selector_ is None:
            raise RuntimeError("StackingSelector has not been fitted.")

        # Build meta-features from fitted base selectors
        meta_features = self._build_meta_features(self.base_selectors_, features)

        # Get predictions from meta-selector
        preds = self.meta_selector_.predict(meta_features)

        if isinstance(preds, dict):
            return cast(
                dict[str, list[tuple[str, int | float]]],
                {str(k): v for k, v in preds.items()},
            )

        # Handle other return types
        result: dict[str, list[tuple[str, float]]] = {}
        for idx in features.index:
            result[str(idx)] = []
        return result

    @staticmethod
    def _define_hyperparameters(
        base_selector_classes: list[type] | None = None,
        meta_selector_classes: list[type] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for StackingSelector.

        Parameters
        ----------
        base_selector_classes : list[type] or None, default=None
            List of base selector classes.
        meta_selector_classes : list[type] or None, default=None
            List of meta selector classes.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters: list[Any] = [
            Categorical("use_generated_features", items=[True, False], default=False),
            Integer("cv", bounds=(2, 10), default=5),
            Categorical("use_original_features", items=[True, False], default=True),
        ]

        if meta_selector_classes:
            hyperparameters.append(
                ClassChoice(
                    "meta_selector",
                    choices=cast(list[type | bool], meta_selector_classes),
                )
            )

        return hyperparameters, [], []
