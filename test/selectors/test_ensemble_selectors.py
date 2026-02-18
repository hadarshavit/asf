"""Tests for ensemble selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asf.selectors import (
    BaggingSelector,
    PerformanceModel,
    StackingSelector,
    VotingSelector,
)
from asf.selectors.multi_class import MultiClassClassifier
from asf.selectors.pairwise_classifier import PairwiseClassifier


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
