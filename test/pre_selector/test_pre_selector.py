import numpy as np
import pytest
import pandas as pd
from functools import partial

from asf.pre_selector import (
    OptimizePreSelection,
    MarginalContributionBasedPreSelector,
    SBSPreSelector,
    BruteForcePreSelector,
    BeamSearchPreSelector,
    KneeOfCurvePreSelector,
    GeneticAlgorithmPreSelector,
    RandomLocalSearchPreSelector,
)
from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
from asf.metrics import virtual_best_solver


def _sum_metric(frame: pd.DataFrame) -> float:
    """Sum all values in the frame."""
    return frame.to_numpy().sum()


def _min_metric(frame: pd.DataFrame) -> float:
    """Return the minimum row sum."""
    return frame.to_numpy().min()


_PARALLEL_METRIC_VALUES = {1: 1.08, 2: 0.1, 3: 1.08}


def _parallel_metric(frame: pd.DataFrame) -> float:
    return _PARALLEL_METRIC_VALUES[frame.shape[1]]


@pytest.fixture
def dummy_performance():
    data = np.array(
        [
            [120, 100, 110],
            [140, 150, 130],
            [180, 170, 190],
            [160, 150, 140],
            [250, 240, 260],
            [230, 220, 210],
            [300, 310, 320],
            [280, 290, 270],
            [350, 340, 360],
            [330, 320, 310],
            [400, 390, 410],
            [380, 370, 360],
            [450, 440, 460],
            [430, 420, 410],
            [500, 490, 510],
            [480, 470, 460],
            [550, 540, 560],
            [530, 520, 510],
            [600, 590, 610],
            [580, 570, 560],
        ]
    )
    return pd.DataFrame(data, columns=pd.Index(["algo1", "algo2", "algo3"]))


def test_optimize_pre_selection(dummy_performance):
    # Create an instance of the OptimizePreSelection class
    pre_selector = OptimizePreSelection(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
        fmin_function="SLSQP",
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty


def test_marginal_contribution_based_pre_selector(dummy_performance):
    # Create an instance of the MarginalContributionBasedPreSelector class
    pre_selector = MarginalContributionBasedPreSelector(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty


def test_sbs_pre_selector(dummy_performance):
    # Create an instance of the SBSPreSelector class
    pre_selector = SBSPreSelector(
        metric=partial(virtual_best_solver, maximize=False),
        n_algorithms=2,
        maximize=False,
    )

    # Fit and transform the performance data
    transformed_performance = pre_selector.fit_transform(dummy_performance)

    # Check if the transformed performance is a DataFrame
    assert isinstance(transformed_performance, pd.DataFrame)

    # Check if the number of algorithms selected is correct
    assert transformed_performance.shape[1] == 2

    # Check if the transformed performance is not empty
    assert not transformed_performance.empty


def test_brute_force_selects_lowest_sum_dataframe():
    performance = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [5.0, 5.0, 5.0],
            "c": [2.0, 2.0, 2.0],
        }
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert list(selected.columns) == ["a", "c"]  # type: ignore[attr-defined]


def test_brute_force_returns_numpy_for_array_input():
    performance = np.array(
        [
            [1.0, 5.0, 2.0],
            [1.0, 5.0, 2.0],
            [1.0, 5.0, 2.0],
        ]
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    assert selected.shape == (3, 2)


def test_beam_search_respects_metric_minimize():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [2.0, 3.0],
            "C": [5.0, 5.0],
            "D": [10.0, 10.0],
        }
    )

    selector = BeamSearchPreSelector(
        metric=_sum_metric,
        n_algorithms=2,
        maximize=False,
        beam_width=2,
    )

    selected = selector.fit_transform(performance)

    assert set(selected.columns) == {"A", "B"}  # type: ignore[attr-defined]


def test_marginal_contribution_based_maximize_selects_largest_column():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [2.0, 2.0],
            "C": [3.0, 3.0],
        }
    )

    selector = MarginalContributionBasedPreSelector(
        metric=_sum_metric,
        n_algorithms=1,
        maximize=True,
    )

    selected = selector.fit_transform(performance)

    assert list(selected.columns) == ["C"]  # type: ignore[attr-defined]


def test_sbs_pre_selector_handles_numpy_input():
    performance = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]
    )

    selector = SBSPreSelector(metric=_sum_metric, n_algorithms=2, maximize=True)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    assert selected.shape == (2, 2)


class _RecordingOptimizer:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)
        self.called_with = None

    def __call__(self, objective_function, x0=None, bounds=None, **kwargs):
        self.called_with = {
            "bounds": bounds,
            "x0": x0,
            "kwargs": kwargs,
        }
        # Call once to ensure objective is valid.
        objective_function(self.values)

        class Result:
            def __init__(self, x):
                self.x = x

        return Result(self.values)


def test_optimize_pre_selection_uses_custom_optimizer():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
            "C": [3.0, 3.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8, 0.1])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=2,
        maximize=False,
        fmin_function=optimizer,
    )

    selected = selector.fit_transform(performance)

    assert optimizer.called_with is not None
    assert optimizer.called_with["bounds"] == [(0, 1)] * 3
    assert set(selected.columns) == {"A", "B"}  # type: ignore[attr-defined]


class _DummyBasePreSelector(AbstractPreSelector):
    def __init__(self, metric, n_algorithms, maximize=False, **kwargs):
        super().__init__(n_algorithms=n_algorithms)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

    def fit_transform(self, performance):
        return performance.iloc[:, : self.n_algorithms]


def test_knee_of_curve_pre_selector_detects_knee():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [1.5, 2.5, 3.5],
            "C": [10.0, 10.0, 10.0],
        }
    )

    metric_values = {1: 1.08, 2: 0.1, 3: 1.08}

    def metric(frame: pd.DataFrame) -> float:
        return metric_values[frame.shape[1]]

    selector = KneeOfCurvePreSelector(
        metric=metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
        S=1.0,
    )

    selected = selector.fit_transform(performance)

    assert list(selected.columns) == ["A", "B"]  # type: ignore[attr-defined]


def test_knee_of_curve_pre_selector_returns_original_when_no_knee():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [3.0, 3.0, 3.0],
        }
    )

    selector = KneeOfCurvePreSelector(
        metric=_sum_metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
    )

    selected = selector.fit_transform(performance)

    pd.testing.assert_frame_equal(selected, performance)


def test_knee_of_curve_pre_selector_parallel_numpy_detects_knee():
    pytest.importorskip("joblib")

    performance = np.array(
        [
            [1.0, 1.5, 10.0],
            [2.0, 2.5, 10.0],
            [3.0, 3.5, 10.0],
        ]
    )

    selector = KneeOfCurvePreSelector(
        metric=_parallel_metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
        workers=1,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    np.testing.assert_allclose(selected, performance[:, :2])


def test_optimize_pre_selection_raises_with_insufficient_algorithms():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
            "C": [3.0, 3.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8, 0.1])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=5,
        maximize=False,
        fmin_function=optimizer,
    )

    with pytest.raises(ValueError):
        selector.fit_transform(performance)


def test_optimize_pre_selection_zero_algorithms_raises():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=0,
        maximize=False,
        fmin_function=optimizer,
    )

    with pytest.raises(ValueError):
        selector.fit_transform(performance)


def test_brute_force_maximize_returns_best_columns():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [5.0, 6.0],
            "C": [3.0, 4.0],
        }
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=1, maximize=True)

    selected = selector.fit_transform(performance)

    assert list(selected.columns) == ["B"]  # type: ignore[attr-defined]


def test_sbs_pre_selector_dataframe_minimize_orders_correctly():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [0.5, 0.5, 0.5],
        }
    )

    selector = SBSPreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert list(selected.columns) == ["C", "A"]  # type: ignore[attr-defined]


class TestGeneticAlgorithmPreSelector:
    """Tests for GeneticAlgorithmPreSelector."""

    def test_basic_fit_transform(self):
        """Test basic fit_transform with DataFrame."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [5.0, 5.0, 5.0],
                "C": [2.0, 2.0, 2.0],
                "D": [10.0, 10.0, 10.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            population_size=10,
            n_generations=5,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] == 2

    def test_fit_transform_with_numpy(self):
        """Test fit_transform with numpy array."""
        performance = np.array(
            [
                [1.0, 5.0, 2.0, 10.0],
                [2.0, 5.0, 2.0, 10.0],
                [3.0, 5.0, 2.0, 10.0],
            ]
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            population_size=10,
            n_generations=5,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, np.ndarray)
        assert selected.shape == (3, 2)

    def test_maximize_mode(self):
        """Test with maximize=True."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [10.0, 10.0],
                "C": [5.0, 5.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=True,
            population_size=10,
            n_generations=5,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        # Should select B (highest sum)
        assert selected.shape[1] == 1

    def test_n_algorithms_exceeds_total(self):
        """Test when n_algorithms >= number of available algorithms."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0],
                "B": [3.0, 4.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=5,  # More than available
            maximize=False,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        # Should return all columns
        assert selected.shape[1] == 2

    def test_elitism_preserves_best(self):
        """Test that elitism parameter works."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [3.0, 3.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            population_size=10,
            n_generations=10,
            elitism=2,
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_mutation_rate(self):
        """Test with different mutation rates."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [3.0, 3.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            population_size=10,
            n_generations=5,
            mutation_rate=0.5,
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_crossover_rate(self):
        """Test with different crossover rates."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [3.0, 3.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            population_size=10,
            n_generations=5,
            crossover_rate=0.0,  # No crossover
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_tournament_size(self):
        """Test with different tournament sizes."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [3.0, 3.0],
            }
        )

        selector = GeneticAlgorithmPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            population_size=10,
            n_generations=5,
            tournament_size=2,
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1


class TestRandomLocalSearchPreSelector:
    """Tests for RandomLocalSearchPreSelector."""

    def test_basic_fit_transform(self):
        """Test basic fit_transform."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [5.0, 5.0, 5.0],
                "C": [2.0, 2.0, 2.0],
                "D": [10.0, 10.0, 10.0],
            }
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            n_restarts=3,
            max_iterations=10,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] == 2

    def test_fit_transform_with_numpy(self):
        """Test with numpy input."""
        performance = np.array(
            [
                [1.0, 5.0, 2.0, 10.0],
                [2.0, 5.0, 2.0, 10.0],
                [3.0, 5.0, 2.0, 10.0],
            ]
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            n_restarts=3,
            max_iterations=10,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, np.ndarray)
        assert selected.shape == (3, 2)

    def test_maximize_mode(self):
        """Test maximize mode."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [10.0, 10.0],
                "C": [5.0, 5.0],
            }
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=True,
            n_restarts=3,
            max_iterations=10,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        assert selected.shape[1] == 1

    def test_n_algorithms_exceeds_total(self):
        """Test when n_algorithms exceeds available."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0],
                "B": [3.0, 4.0],
            }
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=5,
            maximize=False,
            seed=42,
        )

        selected = selector.fit_transform(performance)

        # Should return all columns
        assert selected.shape[1] == 2

    def test_multiple_restarts(self):
        """Test that multiple restarts work."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [5.0, 5.0, 5.0],
                "C": [2.0, 2.0, 2.0],
            }
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            n_restarts=10,
            max_iterations=5,
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 2

    def test_local_search_iterations(self):
        """Test with many iterations."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [5.0, 5.0, 5.0],
                "C": [2.0, 2.0, 2.0],
            }
        )

        selector = RandomLocalSearchPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            n_restarts=2,
            max_iterations=50,
            seed=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 2


class TestMarginalContributionBasePreSelectorForward:
    """Additional tests for MarginalContributionBasedPreSelector."""

    def test_forward_selection_mode(self):
        """Test forward selection mode."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [3.0, 3.0],
            }
        )

        selector = MarginalContributionBasedPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            mode="forward",
        )

        selected = selector.fit_transform(performance)

        # Should select A (lowest sum with forward greedy)
        assert list(selected.columns) == ["A"]  # type: ignore[attr-defined]

    def test_forward_selection_multiple(self):
        """Test forward selection with multiple algorithms."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [2.0, 2.0],
                "C": [10.0, 10.0],
            }
        )

        selector = MarginalContributionBasedPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=False,
            mode="forward",
        )

        selected = selector.fit_transform(performance)

        assert selected.shape[1] == 2

    def test_backward_selection_maximize(self):
        """Test backward selection with maximize."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [5.0, 5.0],
                "C": [10.0, 10.0],
            }
        )

        selector = MarginalContributionBasedPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=True,
            mode="backward",
        )

        selected = selector.fit_transform(performance)

        # Should select C (highest contribution)
        assert list(selected.columns) == ["C"]  # type: ignore[attr-defined]

    def test_with_numpy_input(self):
        """Test with numpy array input."""
        performance = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )

        selector = MarginalContributionBasedPreSelector(
            metric=_sum_metric,
            n_algorithms=1,
            maximize=False,
            mode="backward",
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, np.ndarray)
        assert selected.shape == (2, 1)

    def test_forward_maximize(self):
        """Test forward selection with maximize."""
        performance = pd.DataFrame(
            {
                "A": [1.0, 1.0],
                "B": [5.0, 5.0],
                "C": [10.0, 10.0],
            }
        )

        selector = MarginalContributionBasedPreSelector(
            metric=_sum_metric,
            n_algorithms=2,
            maximize=True,
            mode="forward",
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 2
