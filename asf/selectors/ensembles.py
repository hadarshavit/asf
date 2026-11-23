"""Voting, bagging, stacking, and SAMME boosting for algorithm selection."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold

from asf.metrics.par10 import apply_par10
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.selectors.pairwise_classifier import PairwiseClassifier
from asf.utils.configurable import ClassChoice, ConfigurableMixin
from asf.utils.random_state import random_state_kwargs

try:
    from ConfigSpace import Categorical, Float, Integer

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


Aggregation = Literal["majority", "borda"]
BaggingWeighting = Literal["bootstrap", "full", "out_of_bag"]
SelectorTemplate = AbstractSelector | Callable[..., AbstractSelector]
SelectorPool = Sequence[AbstractSelector] | Callable[..., Sequence[AbstractSelector]]


def _clone_selector(selector: AbstractSelector) -> AbstractSelector:
    # A default constructor would lose the template's configuration.
    return copy.deepcopy(selector)


def _fit_selector(
    template: AbstractSelector,
    features: pd.DataFrame,
    performance: pd.DataFrame,
    fit_kwargs: dict[str, Any],
) -> AbstractSelector:
    fitted = _clone_selector(template)
    fitted.prediction_mode = "aslib"
    fitted.fit(features, performance, **fit_kwargs)
    return fitted


def _training_data(
    owner: AbstractSelector, features: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    if len(features) == 0 or not owner.algorithms:
        raise ValueError("Ensembles require training instances and algorithms")
    if not features.index.equals(performance.index):
        if (
            not performance.index.is_unique
            or not features.index.isin(performance.index).all()
        ):
            raise ValueError(
                "Features and performance must cover the same training instances"
            )
        performance = performance.loc[features.index]
    if len(set(owner.algorithms)) != len(owner.algorithms):
        raise ValueError("Algorithm names must be unique")
    return performance


def _schedules(
    selector: AbstractSelector, features: pd.DataFrame
) -> dict[Any, list[tuple[str, float]]]:
    schedules = selector.predict(features)
    if isinstance(schedules, dict):
        schedules = cast(dict[Any, list[tuple[str, float] | str]], schedules)
        return {
            i: [
                (str(step[0]), step[1])
                for step in schedules.get(i, [])
                if isinstance(step, tuple)
            ]
            for i in features.index
        }
    budget = float(selector.budget or 0)
    if isinstance(schedules, pd.Series):
        return {
            i: [(str(label), budget)] if pd.notna(label) else []
            for i, label in schedules.items()
        }
    # Fitted ensembles from older versions may still have NumPy-mode children.
    return {
        i: [(selector.algorithms[int(row.argmax())], budget)] if row.any() else []
        for i, row in zip(features.index, schedules, strict=True)
    }


def _choice_scores(
    selector: AbstractSelector, features: pd.DataFrame, algorithms: list[str]
) -> pd.DataFrame:
    positions = {algorithm: i for i, algorithm in enumerate(algorithms)}
    scores = np.ones((len(features), len(algorithms)))
    schedules = _schedules(selector, features)
    for i, instance in enumerate(features.index):
        schedule = schedules[instance]
        if not schedule or schedule[0][0] not in positions:
            scores[i] = np.nan
        else:
            scores[i, positions[schedule[0][0]]] = 0
    return pd.DataFrame(scores, index=features.index, columns=pd.Index(algorithms))


def _generated_features(
    selector: AbstractSelector, features: pd.DataFrame
) -> pd.DataFrame | None:
    if not isinstance(selector, AbstractFeatureGenerator):
        return None
    generated_input = features
    if selector.hierarchical_generator is not None:
        generated_input = pd.concat(
            [features, selector.hierarchical_generator.generate_features(features)],
            axis=1,
        )
    generated = selector.generate_features(generated_input)
    if isinstance(generated, pd.DataFrame):
        if generated.index.equals(features.index):
            return generated
        if generated.index.equals(pd.RangeIndex(len(features))):
            return generated.set_axis(features.index)
        return generated.reindex(features.index)
    return pd.DataFrame(generated, index=features.index)


def _ballot(
    selector: AbstractSelector,
    features: pd.DataFrame,
    algorithms: list[str],
    aggregation: Aggregation,
) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((len(features), len(algorithms)))
    scores = _generated_features(selector, features)
    if (
        scores is not None
        and scores.columns.is_unique
        and set(scores.columns) == set(algorithms)
    ):
        values = scores[algorithms].to_numpy(dtype=float, na_value=np.nan)
        if selector.maximize or isinstance(selector, PairwiseClassifier):
            values = -values
    else:
        values = _choice_scores(selector, features, algorithms).to_numpy()
    values = np.where(np.isfinite(values), values, np.inf)
    if aggregation == "borda":
        points = len(algorithms) + 1 - rankdata(values, axis=1, method="average")
        return np.where(np.isfinite(values).any(axis=1, keepdims=True), points, 0)
    return (np.isfinite(values) & (values == values.min(axis=1, keepdims=True))).astype(
        float
    )


def _winners(votes: np.ndarray) -> np.ndarray:
    return np.where(votes.any(axis=1), votes.argmax(axis=1), -1)


def _aggregate(
    owner: AbstractSelector,
    features: pd.DataFrame,
    selectors: Sequence[AbstractSelector],
    weights: Sequence[float],
    aggregation: Aggregation = "majority",
) -> dict[Any, list[tuple[str, float]]]:
    votes = np.zeros((len(features), len(owner.algorithms)))
    for selector, weight in zip(selectors, weights, strict=True):
        if weight:
            votes += weight * _ballot(selector, features, owner.algorithms, aggregation)
    return {
        instance: [(owner.algorithms[winner], float(owner.budget or 0))]
        if winner >= 0
        else []
        for instance, winner in zip(features.index, _winners(votes), strict=True)
    }


def _losses(owner: AbstractSelector, performance: pd.DataFrame) -> np.ndarray:
    values = performance.to_numpy(dtype=float, na_value=np.nan)
    if owner.maximize:
        values = np.where(np.isfinite(values), values, -np.inf)
        best = values.max(axis=1, keepdims=True)
        return np.subtract(
            best, values, out=np.full_like(values, np.inf), where=np.isfinite(best)
        )
    if owner.budget is not None:
        values = np.asarray(apply_par10(values, owner.budget))
    values = np.where(np.isfinite(values), values, np.inf)
    if (values < 0).any():
        values -= values.min()
    return values


def _mean_loss(losses: np.ndarray, choices: np.ndarray) -> float:
    selected = losses[np.arange(len(choices)), choices]
    return float(np.where(choices >= 0, selected, np.inf).mean())


def _normalized_weights(losses: Sequence[float]) -> list[float]:
    weights = 1 / (1 + np.asarray(losses))
    maximum = weights.max()
    return (weights / maximum).tolist() if maximum else [1.0] * len(losses)


def _bootstrap(
    features: pd.DataFrame, performance: pd.DataFrame, positions: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Repeated instances need distinct IDs when a selector indexes by instance.
    return (
        features.iloc[positions].reset_index(drop=True),
        performance.iloc[positions].reset_index(drop=True),
    )


class _EnsembleConfiguration(ConfigurableMixin):
    @classmethod
    def _child_configuration_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        # Constructor arguments for the ensemble can otherwise reach estimators.
        parent_keys = {
            "aggregation",
            "base_selector",
            "base_selector_class",
            "base_selector_classes",
            "base_selectors",
            "budget",
            "cv",
            "feature_groups",
            "feature_selection",
            "hierarchical_generator",
            "maximize",
            "meta_selector",
            "meta_selector_classes",
            "n_estimators",
            "prediction_mode",
            "sample_fraction",
            "use_generated_features",
            "use_original_features",
            "weighting",
        }
        return {key: value for key, value in kwargs.items() if key not in parent_keys}


class BaggingSelector(_EnsembleConfiguration, AbstractSelector):
    """Fit copies of a selector on whole-instance bootstrap samples.

    Majority voting is the default. Use ``aggregation="borda"`` for rank voting.
    Set ``weighting`` to ``"bootstrap"``, ``"full"``, or ``"out_of_bag"`` to
    learn each member's weight on the corresponding training rows.
    """

    PREFIX = "bagging"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selector: SelectorTemplate | None = None,
        n_estimators: int = 10,
        sample_fraction: float = 1.0,
        random_state: int | None = None,
        *,
        aggregation: Aggregation = "majority",
        weighting: BaggingWeighting | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if callable(base_selector) and not isinstance(base_selector, AbstractSelector):
            base_selector = base_selector(
                **random_state_kwargs(base_selector, random_state)
            )
        if base_selector is None:
            raise ValueError("base_selector cannot be None")
        if (
            int(n_estimators) < 1
            or not np.isfinite(sample_fraction)
            or sample_fraction <= 0
        ):
            raise ValueError("n_estimators and sample_fraction must be positive")
        if aggregation not in ("majority", "borda"):
            raise ValueError("aggregation must be 'majority' or 'borda'")
        if weighting not in (None, "bootstrap", "full", "out_of_bag"):
            raise ValueError(
                "weighting must be None, 'bootstrap', 'full', or 'out_of_bag'"
            )

        self.base_selector = base_selector
        self.n_estimators = int(n_estimators)
        self.sample_fraction = float(sample_fraction)
        self.aggregation = aggregation
        self.weighting = weighting
        self.estimators_: list[AbstractSelector] = []
        self.weights_: list[float] = []

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        performance = _training_data(self, features, performance)
        fit_kwargs = dict(kwargs, algorithm_features=self.algorithm_features)
        rng = np.random.default_rng(self.random_state)
        sample_size = max(1, int(len(features) * self.sample_fraction))
        estimators = []
        errors = []
        weighting = getattr(self, "weighting", None)
        losses = _losses(self, performance) if weighting is not None else None

        for _ in range(self.n_estimators):
            positions = rng.choice(len(features), size=sample_size, replace=True)
            x_sample, y_sample = _bootstrap(features, performance, positions)
            estimator = _fit_selector(
                self.base_selector, x_sample, y_sample, fit_kwargs
            )
            estimators.append(estimator)

            if losses is not None:
                evaluation_features = features
                evaluation_losses = losses
                if weighting == "bootstrap":
                    evaluation_features = x_sample
                    evaluation_losses = losses[positions]
                elif weighting == "out_of_bag":
                    held_out = np.ones(len(features), dtype=bool)
                    held_out[positions] = False
                    if held_out.any():
                        evaluation_features = features.iloc[held_out]
                        evaluation_losses = losses[held_out]
                ballot = _ballot(
                    estimator, evaluation_features, self.algorithms, self.aggregation
                )
                errors.append(_mean_loss(evaluation_losses, _winners(ballot)))

        self.estimators_ = estimators
        self.weights_ = (
            _normalized_weights(errors) if errors else [1.0] * len(estimators)
        )

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        if features is None:
            raise ValueError("BaggingSelector requires features for prediction.")
        if not self.estimators_:
            raise RuntimeError("BaggingSelector has not been fitted.")
        weights = getattr(self, "weights_", [1.0] * len(self.estimators_))
        aggregation = getattr(self, "aggregation", "majority")
        return _aggregate(self, features, self.estimators_, weights, aggregation)

    @staticmethod
    def _define_hyperparameters(
        base_selector_class: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []
        params: list[Any] = [
            Integer("n_estimators", bounds=(2, 50), default=10),
            Float("sample_fraction", bounds=(0.5, 1.0), default=1.0),
            Categorical("aggregation", items=["majority", "borda"], default="majority"),
            Categorical("weighting", items=[None, "bootstrap", "full", "out_of_bag"]),
        ]
        if base_selector_class:
            params.append(
                ClassChoice("base_selector", choices=list(base_selector_class))
            )
        return params, [], []


class VotingSelector(ConfigurableMixin, AbstractSelector):
    """Combine selectors trained on the same data using majority or Borda voting.

    Provide fixed ``weights`` or set ``learn_weights=True`` to weight members by
    training performance. Set ``optimize_subset=True`` to search for a better
    subset on the training set. Subset search is exponential in the pool size.
    """

    PREFIX = "voting"
    RETURN_TYPE = "single"

    def __init__(
        self,
        selectors: SelectorPool | None = None,
        weights: list[float] | None = None,
        random_state: int | None = None,
        *,
        aggregation: Aggregation = "majority",
        learn_weights: bool = False,
        optimize_subset: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if selectors is not None and not isinstance(selectors, Sequence):
            selectors = selectors(**random_state_kwargs(selectors, random_state))
        self.selectors = list(selectors) if selectors is not None else []
        if not self.selectors:
            raise ValueError("selectors list cannot be empty")
        if weights is not None and len(weights) != len(self.selectors):
            raise ValueError("weights must have same length as selectors")
        if weights is not None and learn_weights:
            raise ValueError("Use either explicit weights or learn_weights")
        if weights is not None and (not np.isfinite(weights).all() or min(weights) < 0):
            raise ValueError("weights must be finite and nonnegative")
        if aggregation not in ("majority", "borda"):
            raise ValueError("aggregation must be 'majority' or 'borda'")

        self.weights = weights
        self.aggregation = aggregation
        self.learn_weights = learn_weights
        self.optimize_subset = optimize_subset
        self.selectors_: list[AbstractSelector] = []
        self.weights_: list[float] = []
        self.active_selectors_: tuple[int, ...] = ()

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        performance = _training_data(self, features, performance)
        fit_kwargs = dict(kwargs, algorithm_features=self.algorithm_features)
        selectors = [
            _fit_selector(template, features, performance, fit_kwargs)
            for template in self.selectors
        ]
        weights = self.weights if self.weights is not None else [1.0] * len(selectors)
        active = tuple(range(len(selectors)))
        learn_weights = getattr(self, "learn_weights", False)
        optimize_subset = getattr(self, "optimize_subset", False)
        if learn_weights or optimize_subset:
            losses = _losses(self, performance)
            ballots = [
                _ballot(selector, features, self.algorithms, self.aggregation)
                for selector in selectors
            ]
            if learn_weights:
                weights = _normalized_weights(
                    [_mean_loss(losses, _winners(ballot)) for ballot in ballots]
                )
            if optimize_subset:
                weighted = np.asarray(ballots) * np.asarray(weights)[:, None, None]
                best_loss = _mean_loss(losses, _winners(weighted.sum(axis=0)))
                for size in range(len(selectors) - 1, 1, -1):
                    for subset in combinations(range(len(selectors)), size):
                        vote = weighted[list(subset)].sum(axis=0)
                        loss = _mean_loss(losses, _winners(vote))
                        if loss < best_loss:
                            best_loss, active = loss, subset
        self.selectors_, self.weights_, self.active_selectors_ = (
            selectors,
            list(weights),
            active,
        )

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        if features is None:
            raise ValueError("VotingSelector requires features for prediction.")
        if not self.selectors_:
            raise RuntimeError("VotingSelector has not been fitted.")
        weights = getattr(
            self, "weights_", self.weights or [1.0] * len(self.selectors_)
        )
        active = getattr(self, "active_selectors_", tuple(range(len(self.selectors_))))
        return _aggregate(
            self,
            features,
            [self.selectors_[i] for i in active],
            [weights[i] for i in active],
            getattr(self, "aggregation", "majority"),
        )

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []
        return (
            [
                Categorical(
                    "aggregation", items=["majority", "borda"], default="majority"
                ),
                Categorical("learn_weights", items=[False, True]),
                Categorical("optimize_subset", items=[False, True]),
            ],
            [],
            [],
        )


class StackingSelector(_EnsembleConfiguration, AbstractSelector):
    """Train a meta-selector on base predictions and, by default, instance features.

    The default uses full-training predictions like the reference experiment.
    Set ``cv`` to an integer to train the meta-selector on out-of-fold predictions.
    Set ``use_generated_features=False`` to use one-hot choices and
    ``use_original_features=False`` to use prediction features alone.
    ``feature_selection=True`` applies a variance threshold of 0.16.
    """

    PREFIX = "stacking"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selectors: SelectorPool | None = None,
        meta_selector: SelectorTemplate | None = None,
        use_generated_features: bool = True,
        cv: int | None = None,
        use_original_features: bool = True,
        random_state: int | None = None,
        *,
        feature_selection: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if base_selectors is not None and not isinstance(base_selectors, Sequence):
            base_selectors = base_selectors(
                **random_state_kwargs(base_selectors, random_state)
            )
        if callable(meta_selector) and not isinstance(meta_selector, AbstractSelector):
            meta_selector = meta_selector(
                **random_state_kwargs(meta_selector, random_state)
            )
        self.base_selectors = list(base_selectors) if base_selectors is not None else []
        if not self.base_selectors:
            raise ValueError("base_selectors list cannot be empty")
        if meta_selector is None:
            raise ValueError("meta_selector cannot be None")
        if cv is not None and int(cv) < 2:
            raise ValueError("cv must be None or at least two")

        self.meta_selector = meta_selector
        self.use_generated_features = use_generated_features
        self.cv = int(cv) if cv is not None else None
        self.use_original_features = use_original_features
        self.feature_selection = feature_selection
        self.base_selectors_: list[AbstractSelector] = []
        self.meta_selector_: AbstractSelector | None = None

    def _prediction_features(
        self, selectors: Sequence[AbstractSelector], features: pd.DataFrame
    ) -> pd.DataFrame:
        blocks = []
        for i, selector in enumerate(selectors):
            generated = (
                _generated_features(selector, features)
                if self.use_generated_features
                else None
            )
            if generated is None:
                choices = _choice_scores(selector, features, self.algorithms)
                if self.use_generated_features and not hasattr(self, "_encoder"):
                    generated = choices
                else:
                    generated = choices.eq(0).astype(float)
                    # Retain the first-algorithm fallback for hard-choice stacking.
                    generated.loc[choices.isna().all(axis=1), self.algorithms[0]] = 1.0
                    generated.columns = pd.RangeIndex(len(self.algorithms))
            blocks.append(generated.add_prefix(f"sel{i}_"))
        return pd.concat(blocks, axis=1)

    def _meta_features(
        self, features: pd.DataFrame, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        if self.use_original_features:
            return pd.concat([features, predictions], axis=1)
        return predictions

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        performance = _training_data(self, features, performance)
        fit_kwargs = dict(kwargs, algorithm_features=self.algorithm_features)
        if self.cv is not None and len(features) < 2:
            raise ValueError(
                "Out-of-fold stacking requires at least two training instances"
            )
        selectors = [
            _fit_selector(template, features, performance, fit_kwargs)
            for template in self.base_selectors
        ]
        if hasattr(self, "_encoder"):
            del self._encoder
        full_predictions = self._prediction_features(selectors, features)
        if self.cv is None:
            train_predictions = full_predictions
        else:
            folds = KFold(
                n_splits=min(self.cv, len(features)),
                shuffle=True,
                random_state=self.random_state,
            )
            blocks = []
            for train, validation in folds.split(features):
                fold_selectors = [
                    _fit_selector(
                        template,
                        features.iloc[train],
                        performance.iloc[train],
                        fit_kwargs,
                    )
                    for template in self.base_selectors
                ]
                predictions = self._prediction_features(
                    fold_selectors, features.iloc[validation]
                )
                blocks.append(predictions.set_axis(validation))
            train_predictions = pd.concat(blocks).sort_index().set_axis(features.index)
            train_predictions = train_predictions.reindex(
                columns=full_predictions.columns
            )

        prediction_columns = full_predictions.columns
        meta_features = self._meta_features(features, train_predictions.fillna(0))
        feature_filter = None
        if getattr(self, "feature_selection", False):
            feature_filter = VarianceThreshold(threshold=0.16)
            feature_filter.fit(meta_features.to_numpy())
            meta_features = meta_features.iloc[:, feature_filter.get_support()]

        self.meta_selector_ = _fit_selector(
            self.meta_selector, meta_features, performance, fit_kwargs
        )
        self.base_selectors_ = selectors
        self._prediction_columns = prediction_columns
        self._feature_filter = feature_filter

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        if features is None:
            raise ValueError("StackingSelector requires features for prediction.")
        if not self.base_selectors_ or self.meta_selector_ is None:
            raise RuntimeError("StackingSelector has not been fitted.")
        if len(features) == 0:
            return {}

        predictions = self._prediction_features(self.base_selectors_, features)
        if hasattr(self, "_prediction_columns"):
            predictions = predictions.reindex(columns=self._prediction_columns).fillna(
                0
            )
        meta_features = self._meta_features(features, predictions)
        feature_filter = getattr(self, "_feature_filter", None)
        if feature_filter is not None:
            meta_features = meta_features.iloc[:, feature_filter.get_support()]

        return _schedules(self.meta_selector_, meta_features)

    @staticmethod
    def _define_hyperparameters(
        base_selector_classes: list[type] | None = None,
        meta_selector_classes: list[type] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []
        params: list[Any] = [
            Categorical("use_generated_features", items=[True, False]),
            Categorical("cv", items=[None, *range(2, 11)]),
            Categorical("use_original_features", items=[True, False]),
            Categorical("feature_selection", items=[False, True]),
        ]
        if meta_selector_classes:
            params.append(
                ClassChoice("meta_selector", choices=list(meta_selector_classes))
            )
        return params, [], []


class BoostingSelector(_EnsembleConfiguration, AbstractSelector):
    """Boost a base selector using multiclass SAMME and weighted resampling.

    The first selector trains on the full data. Later members train on
    full-size bootstrap samples drawn with higher weight on mistakes.
    """

    PREFIX = "boosting"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selector: SelectorTemplate,
        n_estimators: int = 10,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if callable(base_selector) and not isinstance(base_selector, AbstractSelector):
            base_selector = base_selector(
                **random_state_kwargs(base_selector, random_state)
            )
        if base_selector is None:
            raise ValueError("base_selector cannot be None")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be positive")

        self.base_selector = base_selector
        self.n_estimators = int(n_estimators)
        self.estimators_: list[AbstractSelector] = []
        self.weights_: list[float] = []
        self._fitted = False

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        performance = _training_data(self, features, performance)
        values = performance.to_numpy(dtype=float, na_value=np.nan)
        if self.maximize:
            values = -values
        values = np.where(np.isfinite(values), values, np.inf)
        optimal = np.isfinite(values) & (values == values.min(axis=1, keepdims=True))
        eligible = optimal.any(axis=1)
        n_rows, n_algorithms = optimal.shape
        if not eligible.any() and n_algorithms > 1:
            raise ValueError(
                "BoostingSelector requires at least one observed performance value"
            )

        estimators: list[AbstractSelector] = []
        weights: list[float] = []
        if n_algorithms > 1:
            fit_kwargs = dict(kwargs, algorithm_features=self.algorithm_features)
            rng = np.random.default_rng(self.random_state)
            probabilities = eligible.astype(float) / eligible.sum()
            for step in range(self.n_estimators):
                if step == 0:
                    x_sample, y_sample = features, performance
                else:
                    positions = rng.choice(
                        n_rows, size=n_rows, replace=True, p=probabilities
                    )
                    x_sample, y_sample = _bootstrap(features, performance, positions)
                estimator = _fit_selector(
                    self.base_selector, x_sample, y_sample, fit_kwargs
                )
                choices = _winners(
                    _ballot(estimator, features, self.algorithms, "majority")
                )
                wrong = (choices < 0) | ~optimal[np.arange(n_rows), choices]
                error = float(probabilities[wrong].sum())
                if error >= 1 - 1 / n_algorithms:
                    break

                estimators.append(estimator)
                if error == 0:
                    weights.append(sum(weights) + 1)
                    break

                weight = float(
                    np.log1p(-error) - np.log(error) + np.log(n_algorithms - 1)
                )
                weights.append(weight)
                # Downweight correct rows to avoid overflow before normalization.
                probabilities[~wrong] *= np.exp(-weight)
                probabilities /= probabilities.sum()

        self.estimators_, self.weights_ = estimators, weights
        self._fitted = True

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        if features is None:
            raise ValueError("BoostingSelector requires features for prediction.")
        if not self._fitted:
            raise RuntimeError("BoostingSelector has not been fitted.")
        votes = np.zeros((len(features), len(self.algorithms)))
        for estimator, weight in zip(self.estimators_, self.weights_, strict=True):
            choices = _winners(
                _ballot(estimator, features, self.algorithms, "majority")
            )
            valid = choices >= 0
            votes[np.flatnonzero(valid), choices[valid]] += weight
        return {
            instance: [(self.algorithms[winner], float(self.budget or 0))]
            for instance, winner in zip(
                features.index, votes.argmax(axis=1), strict=True
            )
        }

    @staticmethod
    def _define_hyperparameters(
        base_selector_class: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []
        params: list[Any] = [Integer("n_estimators", bounds=(1, 50), default=10)]
        if base_selector_class:
            params.append(
                ClassChoice("base_selector", choices=list(base_selector_class))
            )
        return params, [], []
