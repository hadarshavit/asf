"""Tests for ensemble selectors."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from asf.selectors import (
    AbstractFeatureGenerator,
    AbstractSelector,
    BaggingSelector,
    BoostingSelector,
    PerformanceModel,
    StackingSelector,
    VotingSelector,
)
from asf.selectors.ensembles import Aggregation
from asf.selectors.multi_class import MultiClassClassifier
from asf.selectors.pairwise_classifier import PairwiseClassifier


class RecordedSelector(AbstractSelector):
    """Choose by training membership or use a fixed choice for every row."""

    fit_count = 0

    def __init__(self, selected: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.selected = selected

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        type(self).fit_count += 1
        self.training_features = features.copy()
        self.training_performance = performance.copy()
        self.fit_kwargs = kwargs

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        assert features is not None
        assert features.columns.equals(self.training_features.columns)
        trained = set(self.training_features.iloc[:, 0])
        return {
            instance: [(self.selected or ("A" if row in trained else "B"), 0.0)]
            for instance, row in features.iloc[:, 0].items()
        }


class FixedScores(RecordedSelector, AbstractFeatureGenerator):
    def __init__(self, scores: list[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scores = scores

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [self.scores] * len(base_features),
            index=base_features.index,
            columns=pd.Index(self.algorithms),
        )


class BoostingMember(RecordedSelector):
    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[Any, list[tuple[str, float]]]:
        assert features is not None
        return {
            instance: [
                (
                    "C"
                    if x == 3
                    and 3 in set(self.training_features["x"])
                    and isinstance(self.training_features.index, pd.RangeIndex)
                    else "A",
                    0.0,
                )
            ]
            for instance, x in features["x"].items()
        }


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample training data."""
    np.random.seed(42)
    n_instances = 50
    n_features = 5
    n_algorithms = 4

    features = pd.DataFrame(
        np.random.randn(n_instances, n_features),
        columns=[f"f_{i}" for i in range(n_features)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    performance = pd.DataFrame(
        np.random.rand(n_instances, n_algorithms) * 100,
        columns=[f"algo_{i}" for i in range(n_algorithms)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    return features, performance


@pytest.fixture
def test_features() -> pd.DataFrame:
    """Create test features."""
    np.random.seed(123)
    return pd.DataFrame(
        np.random.randn(10, 5),
        columns=[f"f_{i}" for i in range(5)],
        index=[f"test_{i}" for i in range(10)],
    )


class TestBaggingSelector:
    """Tests for BaggingSelector."""

    def test_init_with_selector(self) -> None:
        """Test initialization with a base selector."""
        base = PerformanceModel()
        bagging = BaggingSelector(base_selector=base, n_estimators=5)

        assert bagging.base_selector is base
        assert bagging.n_estimators == 5
        assert bagging.sample_fraction == 1.0

    def test_init_with_callable(self) -> None:
        """Test initialization with a callable."""
        bagging = BaggingSelector(
            base_selector=lambda: PerformanceModel(),
            n_estimators=3,
        )
        assert isinstance(bagging.base_selector, PerformanceModel)

    def test_init_requires_base_selector(self) -> None:
        """Test that None base_selector raises ValueError."""
        with pytest.raises(ValueError, match="base_selector cannot be None"):
            BaggingSelector(base_selector=None)

    def test_fit_and_predict(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test fitting and prediction."""
        features, performance = sample_data

        bagging = BaggingSelector(
            base_selector=PerformanceModel(),
            n_estimators=3,
            random_state=42,
        )

        bagging.fit(features, performance)

        assert len(bagging.estimators_) == 3

        preds = bagging.predict(test_features)
        assert isinstance(preds, dict)
        preds = cast(dict[str, list[tuple[str, float]]], preds)
        assert len(preds) == len(test_features)

        for instance, schedule in preds.items():
            assert isinstance(instance, str)
            assert isinstance(schedule, list)
            if schedule and len(schedule) > 0:
                assert len(schedule[0]) == 2  # (algo, budget)

    def test_bootstrap_sampling(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that different bootstrap samples are used."""
        features, performance = sample_data

        bagging = BaggingSelector(
            base_selector=PerformanceModel(),
            n_estimators=5,
            sample_fraction=0.8,
            random_state=42,
        )

        bagging.fit(features, performance)

        # All estimators should be fitted
        assert all(
            hasattr(est, "algorithms") and est.algorithms for est in bagging.estimators_
        )

    def test_predict_requires_features(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test that predict raises error without features."""
        features, performance = sample_data

        bagging = BaggingSelector(
            base_selector=PerformanceModel(),
            n_estimators=2,
        )
        bagging.fit(features, performance)

        with pytest.raises(ValueError, match="requires features"):
            bagging.predict(None)


class TestVotingSelector:
    """Tests for VotingSelector."""

    def test_init_with_selectors(self) -> None:
        """Test initialization with selector list."""
        selectors = [PerformanceModel(), PairwiseClassifier()]
        voting = VotingSelector(selectors=selectors)

        assert len(voting.selectors) == 2

    def test_init_with_callable(self) -> None:
        """Test initialization with callable."""
        voting = VotingSelector(
            selectors=lambda: [PerformanceModel(), MultiClassClassifier()]
        )
        assert len(voting.selectors) == 2

    def test_init_requires_selectors(self) -> None:
        """Test that empty selectors raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            VotingSelector(selectors=[])

    def test_weights_validation(self) -> None:
        """Test that weights must match selectors length."""
        with pytest.raises(ValueError, match="same length"):
            VotingSelector(
                selectors=[PerformanceModel(), MultiClassClassifier()],
                weights=[1.0],
            )

    def test_fit_and_predict(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test fitting and prediction."""
        features, performance = sample_data

        voting = VotingSelector(selectors=[PerformanceModel(), MultiClassClassifier()])

        voting.fit(features, performance)

        assert len(voting.selectors_) == 2

        preds = voting.predict(test_features)
        assert isinstance(preds, dict)
        assert len(preds) == len(test_features)

    def test_weighted_voting(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test weighted voting."""
        features, performance = sample_data

        voting = VotingSelector(
            selectors=[PerformanceModel(), MultiClassClassifier()],
            weights=[2.0, 1.0],
        )

        voting.fit(features, performance)
        preds = voting.predict(test_features)
        assert isinstance(preds, dict)

        # Should return valid predictions
        assert all(isinstance(v, list) for v in preds.values())


class TestStackingSelector:
    """Tests for StackingSelector."""

    def test_init_with_selectors(self) -> None:
        """Test initialization."""
        stacking = StackingSelector(
            base_selectors=[PerformanceModel()],
            meta_selector=MultiClassClassifier(),
        )

        assert len(stacking.base_selectors) == 1
        assert isinstance(stacking.meta_selector, MultiClassClassifier)

    def test_init_requires_base_selectors(self) -> None:
        """Test that empty base_selectors raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StackingSelector(
                base_selectors=[],
                meta_selector=MultiClassClassifier(),
            )

    def test_init_requires_meta_selector(self) -> None:
        """Test that None meta_selector raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            StackingSelector(
                base_selectors=[PerformanceModel()],
                meta_selector=None,
            )

    def test_fit_and_predict_without_generated_features(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test stacking with one-hot encoded predictions."""
        features, performance = sample_data

        stacking = StackingSelector(
            base_selectors=[PerformanceModel(), MultiClassClassifier()],
            meta_selector=PerformanceModel(),
            use_generated_features=False,
            cv=3,
            random_state=42,
        )

        stacking.fit(features, performance)

        assert len(stacking.base_selectors_) == 2
        assert stacking.meta_selector_ is not None

        preds = stacking.predict(test_features)
        assert isinstance(preds, dict)
        assert len(preds) == len(test_features)

    def test_fit_and_predict_with_generated_features(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test stacking with generated features."""
        features, performance = sample_data

        # PerformanceModel has generate_features
        stacking = StackingSelector(
            base_selectors=[PerformanceModel(), PairwiseClassifier()],
            meta_selector=MultiClassClassifier(),
            use_generated_features=True,
            cv=3,
            random_state=42,
        )

        stacking.fit(features, performance)
        preds = stacking.predict(test_features)

        assert isinstance(preds, dict)
        assert len(preds) == len(test_features)

    def test_use_original_features_true(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test that original features are included."""
        features, performance = sample_data

        stacking = StackingSelector(
            base_selectors=[PerformanceModel()],
            meta_selector=MultiClassClassifier(),
            use_original_features=True,
            cv=3,
            random_state=42,
        )

        stacking.fit(features, performance)
        preds = stacking.predict(test_features)

        # Should work and produce predictions
        assert len(preds) == len(test_features)

    def test_use_original_features_false(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test stacking without original features."""
        features, performance = sample_data

        stacking = StackingSelector(
            base_selectors=[PerformanceModel()],
            meta_selector=MultiClassClassifier(),
            use_original_features=False,
            cv=3,
            random_state=42,
        )

        stacking.fit(features, performance)
        preds = stacking.predict(test_features)

        assert len(preds) == len(test_features)

    def test_cv_parameter(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test that cv parameter is respected."""
        features, performance = sample_data

        for cv in [2, 5]:
            stacking = StackingSelector(
                base_selectors=[PerformanceModel()],
                meta_selector=MultiClassClassifier(),
                cv=cv,
                random_state=42,
            )

            # Should fit without error
            stacking.fit(features, performance)
            assert stacking.cv == cv


class TestEnsembleIntegration:
    """Integration tests for ensemble selectors."""

    def test_nested_ensembles(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test nesting ensemble selectors."""
        features, performance = sample_data

        # Bagging inside Voting
        bagging1 = BaggingSelector(
            base_selector=PerformanceModel(),
            n_estimators=2,
            random_state=42,
        )
        bagging2 = BaggingSelector(
            base_selector=MultiClassClassifier(),
            n_estimators=2,
            random_state=43,
        )

        voting = VotingSelector(selectors=[bagging1, bagging2])
        voting.fit(features, performance)
        preds = voting.predict(test_features)

        assert len(preds) == len(test_features)

    def test_stacking_with_bagging(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
        test_features: pd.DataFrame,
    ) -> None:
        """Test stacking with bagging as base selector."""
        features, performance = sample_data

        bagging = BaggingSelector(
            base_selector=PerformanceModel(),
            n_estimators=2,
            random_state=42,
        )

        stacking = StackingSelector(
            base_selectors=[bagging, MultiClassClassifier()],
            meta_selector=PerformanceModel(),
            cv=2,
            random_state=42,
        )

        stacking.fit(features, performance)
        preds = stacking.predict(test_features)

        assert len(preds) == len(test_features)


def test_reference_aggregation_modes_choose_different_winners() -> None:
    features = pd.DataFrame({"x": [0]}, index=pd.Index([42]))
    performance = pd.DataFrame(
        [[20, 5, 0, 11]], columns=pd.Index(list("ABCD")), index=features.index
    )
    members = [
        FixedScores([0, 1, 2, 3]),
        FixedScores([0, 1, 2, 3]),
        FixedScores([3, 0, 2, 1]),
        FixedScores([3, 1, 0, 2]),
    ]
    majority = VotingSelector(members, budget=10)
    borda = VotingSelector(members, aggregation="borda", budget=10)
    weighted = VotingSelector(members, learn_weights=True, budget=10)

    for ensemble, winner in [(majority, "A"), (borda, "B"), (weighted, "C")]:
        ensemble.fit(features, performance)
        assert ensemble.predict(features) == {42: [(winner, 10.0)]}
    assert weighted.weights_ == pytest.approx([1 / 101, 1 / 101, 1 / 6, 1])
    assert not hasattr(members[0], "training_features")


@pytest.mark.parametrize("aggregation", ["majority", "borda"])
def test_aggregation_uses_portfolio_order_for_tied_scores(
    aggregation: Aggregation,
) -> None:
    features = pd.DataFrame({"x": [0]})
    performance = pd.DataFrame({"B": [1], "A": [2], "C": [3]})
    ensemble = VotingSelector(
        [FixedScores([2, 2, 1], maximize=True)],
        aggregation=aggregation,
    )
    ensemble.fit(features, performance)
    assert ensemble.predict(features) == {0: [("B", 0.0)]}


def test_voting_selects_best_subset_and_resets_on_refit() -> None:
    features = pd.DataFrame({"x": [0]})
    ensemble = VotingSelector(
        [RecordedSelector("A"), RecordedSelector("B"), RecordedSelector("A")],
        optimize_subset=True,
    )
    ensemble.fit(features, pd.DataFrame({"B": [1], "A": [10]}))
    assert ensemble.active_selectors_ == (0, 1)
    assert ensemble.predict(features) == {0: [("B", 0.0)]}

    ensemble.fit(features, pd.DataFrame({"B": [10], "A": [1]}))
    assert ensemble.active_selectors_ == (0, 1, 2)
    assert ensemble.predict(features) == {0: [("A", 0.0)]}


def test_bagging_weights_use_out_of_bag_rows_and_aligned_training_data() -> None:
    features = pd.DataFrame({"x": [0, 1, 2, 3]}, index=pd.Index(list("wyzx")))
    performance = pd.DataFrame({"A": features.x, "B": 10 * features.x})
    algorithm_features = pd.DataFrame({"config": [0, 1]}, index=pd.Index(list("AB")))
    ensemble = BaggingSelector(
        RecordedSelector(),
        n_estimators=2,
        random_state=0,
        weighting="out_of_bag",
        aggregation="borda",
    )
    ensemble.fit(
        features,
        performance.iloc[::-1],
        algorithm_features=algorithm_features,
        flag=True,
    )
    losses = []
    for member in ensemble.estimators_:
        assert isinstance(member, RecordedSelector)
        rows = member.training_features.x
        pd.testing.assert_series_equal(
            rows, member.training_performance.A, check_names=False
        )
        pd.testing.assert_frame_equal(member.algorithm_features, algorithm_features)
        assert member.fit_kwargs == {"flag": True}
        losses.append(performance.loc[~features.x.isin(rows), "B"].mean())
    assert len(set(losses)) == 2
    assert ensemble.weights_ == pytest.approx(
        (1 + min(losses)) / (1 + np.array(losses))
    )
    predictions = ensemble.predict(features)
    assert isinstance(predictions, dict) and all(predictions.values())


@pytest.mark.parametrize("cv, expected_fits", [(None, 2), (2, 4)])
def test_stacking_reference_defaults_and_optional_out_of_fold(
    cv: int | None, expected_fits: int
) -> None:
    features = pd.DataFrame(
        {"x": [0, 1, 2, 3], "constant": [0] * 4}, index=pd.Index([8, 4, 2, 6])
    )
    performance = pd.DataFrame({"A": [1] * 4, "B": [2] * 4}, index=features.index)
    before = RecordedSelector.fit_count
    stacking = StackingSelector([RecordedSelector()], RecordedSelector("A"), cv=cv)
    stacking.fit(features, performance)

    assert RecordedSelector.fit_count - before == expected_fits
    meta = stacking.meta_selector_
    assert isinstance(meta, RecordedSelector)
    assert list(meta.training_features.columns) == ["x", "constant", "sel0_A", "sel0_B"]
    assert list(meta.training_features.sel0_A) == [int(cv is not None)] * len(features)
    assert list(stacking.predict(features)) == list(features.index)

    filtered = StackingSelector(
        [FixedScores([4, 9])], RecordedSelector("A"), feature_selection=True
    )
    filtered.fit(features, performance)
    assert isinstance(filtered.meta_selector_, RecordedSelector)
    assert list(filtered.meta_selector_.training_features.columns) == ["x"]
    assert list(filtered.predict(features)) == list(features.index)


def test_samme_reweights_mistakes_and_stops_on_perfect_member() -> None:
    features = pd.DataFrame({"x": [0, 1, 2, 3]}, index=pd.Index(list("wxyz")))
    performance = pd.DataFrame(
        {"A": [0, 0, 0, 2], "B": [0, 1, 1, 1], "C": [1, 1, 1, 0]},
        index=features.index,
    )
    ensemble = BoostingSelector(
        BoostingMember(), n_estimators=5, random_state=42, budget=10
    )
    ensemble.fit(features, performance)

    assert ensemble.weights_ == pytest.approx([np.log(6), np.log(6) + 1])
    first, second = ensemble.estimators_
    assert isinstance(first, BoostingMember) and isinstance(second, BoostingMember)
    pd.testing.assert_frame_equal(first.training_features, features)
    expected = np.random.default_rng(42).choice(4, size=4, p=[1 / 9] * 3 + [2 / 3])
    assert list(second.training_features.x) == list(expected)
    assert ensemble.predict(features)["z"] == [("C", 10.0)]


@pytest.mark.parametrize(
    "performance, chosen, expected_count",
    [
        ({"A": [1, 1], "B": [0, 0]}, "A", 0),
        ({"A": [1, 1], "B": [0, 0]}, "B", 1),
        ({"A": [1, 1]}, "B", 0),
    ],
)
def test_samme_edge_cases(
    performance: dict[str, list[int]], chosen: str, expected_count: int
) -> None:
    features = pd.DataFrame({"x": [0, 1]})
    ensemble = BoostingSelector(RecordedSelector(chosen), prediction_mode="pandas")
    ensemble.fit(features, pd.DataFrame(performance))
    assert len(ensemble.estimators_) == expected_count
    assert list(ensemble.predict(features)) == [chosen if expected_count else "A"] * 2


@pytest.mark.parametrize(
    "selector_class, space_kwargs, instance_kwargs",
    [
        (
            BaggingSelector,
            {"base_selector_class": [PerformanceModel]},
            {"n_estimators": 2},
        ),
        (
            BoostingSelector,
            {"base_selector_class": [PerformanceModel]},
            {"n_estimators": 2},
        ),
        (
            StackingSelector,
            {"meta_selector_classes": [PerformanceModel]},
            {"base_selectors": [PerformanceModel()]},
        ),
        (VotingSelector, {}, {"selectors": [PerformanceModel()]}),
    ],
)
def test_ensemble_default_configuration_fits_nested_models(
    selector_class: type[
        BaggingSelector | BoostingSelector | StackingSelector | VotingSelector
    ],
    space_kwargs: dict[str, Any],
    instance_kwargs: dict[str, Any],
) -> None:
    pytest.importorskip("ConfigSpace")
    features = np.arange(24.0).reshape(12, 2)
    performance = np.column_stack((features[:, 0] + 1, 25 - features[:, 0]))
    space = selector_class.get_configuration_space(**space_kwargs)
    config = space.get_default_configuration()
    params = dict(space_kwargs)
    if "base_selectors" in instance_kwargs:
        params["base_selectors"] = instance_kwargs["base_selectors"]
    configured = selector_class.get_from_configuration(
        config, budget=100, prediction_mode="numpy", **params
    )
    selector = configured(**instance_kwargs)
    selector.fit(features, performance)
    predictions = selector.predict(features[:2])
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2, 2)
