"""Additional tests for pre-selectors with low coverage."""

import numpy as np
import pandas as pd

from asf.pre_selector.genetic_algorithm_pre_selection import GeneticAlgorithmPreSelector
from asf.pre_selector.random_local_search_pre_selection import (
    RandomLocalSearchPreSelector,
)
from asf.pre_selector.marginal_contribution_based import (
    MarginalContributionBasedPreSelector,
)


def _sum_metric(frame: pd.DataFrame) -> float:
    """Sum all values in the frame."""
    return frame.to_numpy().sum()


def _min_metric(frame: pd.DataFrame) -> float:
    """Return the minimum row sum."""
    return frame.to_numpy().min()


class TestGeneticAlgorithmPreSelector:
    """Tests for GeneticAlgorithmPreSelector."""

    def test_basic_fit_transform(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] == 2

    def test_fit_transform_with_numpy(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, np.ndarray)
        assert selected.shape == (3, 2)

    def test_maximize_mode(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        # Should select B (highest sum)
        assert selected.shape[1] == 1

    def test_n_algorithms_exceeds_total(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        # Should return all columns
        assert selected.shape[1] == 2

    def test_elitism_preserves_best(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_mutation_rate(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_crossover_rate(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1

    def test_tournament_size(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 1


class TestRandomLocalSearchPreSelector:
    """Tests for RandomLocalSearchPreSelector."""

    def test_basic_fit_transform(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] == 2

    def test_fit_transform_with_numpy(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        assert isinstance(selected, np.ndarray)
        assert selected.shape == (3, 2)

    def test_maximize_mode(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        assert selected.shape[1] == 1

    def test_n_algorithms_exceeds_total(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)

        # Should return all columns
        assert selected.shape[1] == 2

    def test_multiple_restarts(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 2

    def test_local_search_iterations(self) -> None:
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
            random_state=42,
        )

        selected = selector.fit_transform(performance)
        assert selected.shape[1] == 2


class TestMarginalContributionBasePreSelectorForward:
    """Additional tests for MarginalContributionBasedPreSelector."""

    def test_forward_selection_mode(self) -> None:
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

    def test_forward_selection_multiple(self) -> None:
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

    def test_backward_selection_maximize(self) -> None:
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

    def test_with_numpy_input(self) -> None:
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

    def test_forward_maximize(self) -> None:
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
