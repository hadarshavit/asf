"""Tests for SATzilla pre-solver implementations."""

import numpy as np
import pandas as pd
import pytest

from asf.presolving.satzilla_presolver import (
    SATzillaPresolver,
    SATzillaManualPresolver,
    SATzillaHybridPresolver,
)


@pytest.fixture
def dummy_data():
    """Create test data with diverse solver performance."""
    rng = np.random.RandomState(42)
    n_instances = 100
    n_algorithms = 5

    # Create performance data (runtimes)
    performance = pd.DataFrame(
        rng.exponential(scale=10, size=(n_instances, n_algorithms)),
        columns=[f"algo_{i}" for i in range(n_algorithms)],
    )

    # Make some algorithms solve some instances very fast
    # algo_0: solves 30% of instances in 1 second
    performance["algo_0"] = np.where(rng.rand(n_instances) < 0.3, 1.0, 100.0)
    # algo_1: solves 40% of instances in 2 seconds
    performance["algo_1"] = np.where(rng.rand(n_instances) < 0.4, 2.0, 50.0)

    features = pd.DataFrame(rng.randn(n_instances, 10))

    return features, performance


@pytest.fixture
def simple_data():
    """Create simple test data."""
    rng = np.random.RandomState(0)
    features = pd.DataFrame(rng.randn(20, 3), columns=["f1", "f2", "f3"])
    performance = pd.DataFrame(
        rng.exponential(15, (20, 4)), columns=["algo1", "algo2", "algo3", "algo4"]
    )
    return features, performance


class TestSATzillaPresolver:
    """Tests for the automatic SATzilla presolver."""

    def test_basic_fit_and_predict(self, dummy_data):
        """Test basic fit and predict flow."""
        features, performance = dummy_data

        presolver = SATzillaPresolver(budget=15.0, max_presolvers=2)
        presolver.fit(features, performance)
        schedule = presolver.predict()

        assert isinstance(schedule, list)
        for item in schedule:
            assert isinstance(item, tuple)
            assert len(item) == 2
            algo, cutoff = item
            assert isinstance(algo, str)
            assert isinstance(cutoff, (int, float))
            assert cutoff > 0

    def test_budget_constraint(self, dummy_data):
        """Test that schedule respects budget constraint."""
        features, performance = dummy_data
        budget = 10.0

        presolver = SATzillaPresolver(budget=budget, max_presolvers=2)
        presolver.fit(features, performance)
        schedule = presolver.predict()

        total_time = sum(cutoff for _, cutoff in schedule)
        assert total_time <= budget

    def test_empty_schedule_possible(self, simple_data):
        """Test that empty schedule can be returned if no pre-solver helps."""
        features, performance = simple_data

        # Very small budget that may not allow any meaningful presolver
        presolver = SATzillaPresolver(budget=0.5, candidate_cutoffs=[0.5, 1.0])
        presolver.fit(features, performance)
        schedule = presolver.predict()

        # Schedule may be empty or have items with cutoff <= budget
        total_time = sum(cutoff for _, cutoff in schedule)
        assert total_time <= 0.5

    def test_single_presolver(self, dummy_data):
        """Test with max_presolvers=1."""
        features, performance = dummy_data

        presolver = SATzillaPresolver(budget=10.0, max_presolvers=1)
        presolver.fit(features, performance)
        schedule = presolver.predict()

        assert len(schedule) <= 1

    def test_selection_stats(self, dummy_data):
        """Test that selection statistics are available."""
        features, performance = dummy_data

        presolver = SATzillaPresolver(budget=15.0)
        presolver.fit(features, performance)
        stats = presolver.get_selection_stats()

        assert "group1_candidates" in stats
        assert "n_configurations_evaluated" in stats
        assert "best_solved" in stats
        assert stats["n_configurations_evaluated"] > 0

    def test_with_solver_groups(self, dummy_data):
        """Test with explicit solver groups."""
        features, performance = dummy_data

        presolver = SATzillaPresolver(
            budget=15.0,
            group1_solvers=["algo_0", "algo_2"],
            group2_solvers=["algo_1", "algo_3"],
            auto_select_candidates=False,
        )
        presolver.fit(features, performance)
        schedule = presolver.predict()

        # All algorithms in schedule should be from the specified groups
        valid_algos = {"algo_0", "algo_1", "algo_2", "algo_3"}
        for algo, _ in schedule:
            assert algo in valid_algos

    def test_custom_cutoffs(self, dummy_data):
        """Test with custom cutoff times."""
        features, performance = dummy_data
        custom_cutoffs = [1.0, 3.0, 7.0]

        presolver = SATzillaPresolver(budget=15.0, candidate_cutoffs=custom_cutoffs)
        presolver.fit(features, performance)
        schedule = presolver.predict()

        # All cutoffs should be from custom_cutoffs
        for _, cutoff in schedule:
            assert cutoff in custom_cutoffs


class TestSATzillaManualPresolver:
    """Tests for the manual SATzilla presolver."""

    def test_basic_usage(self, dummy_data):
        """Test basic manual presolver usage."""
        features, performance = dummy_data
        schedule = [("algo_0", 5.0), ("algo_1", 2.0)]

        presolver = SATzillaManualPresolver(schedule=schedule, budget=15.0)
        presolver.fit(features, performance)
        result = presolver.predict()

        assert result == schedule

    def test_budget_validation(self):
        """Test that budget constraint is validated."""
        with pytest.raises(ValueError):
            SATzillaManualPresolver(
                schedule=[("algo_0", 10.0), ("algo_1", 10.0)], budget=15.0
            )

    def test_algorithm_validation(self, dummy_data):
        """Test that algorithm existence is validated during fit."""
        features, performance = dummy_data

        presolver = SATzillaManualPresolver(
            schedule=[("nonexistent_algo", 5.0)], budget=15.0
        )

        with pytest.raises(ValueError):
            presolver.fit(features, performance)

    def test_empty_schedule(self, dummy_data):
        """Test with empty schedule."""
        features, performance = dummy_data

        presolver = SATzillaManualPresolver(schedule=[], budget=15.0)
        presolver.fit(features, performance)
        result = presolver.predict()

        assert result == []


class TestSATzillaHybridPresolver:
    """Tests for the hybrid SATzilla presolver."""

    def test_basic_hybrid(self, dummy_data):
        """Test basic hybrid presolver."""
        features, performance = dummy_data

        presolver = SATzillaHybridPresolver(
            budget=15.0,
            complete_solvers=["algo_0", "algo_2", "algo_4"],
            local_search_solvers=["algo_1", "algo_3"],
        )
        presolver.fit(features, performance)
        schedule = presolver.predict()

        assert isinstance(schedule, list)
        total_time = sum(cutoff for _, cutoff in schedule)
        assert total_time <= 15.0

    def test_hybrid_selection_stats(self, dummy_data):
        """Test that hybrid presolver provides selection stats."""
        features, performance = dummy_data

        presolver = SATzillaHybridPresolver(
            budget=15.0,
            complete_solvers=["algo_0", "algo_2"],
            local_search_solvers=["algo_1", "algo_3"],
        )
        presolver.fit(features, performance)
        stats = presolver.get_selection_stats()

        assert "group1_candidates" in stats
        assert "group2_candidates" in stats
        # Should have candidates from both groups
        assert len(stats["group1_candidates"]) > 0
        assert len(stats["group2_candidates"]) > 0

    def test_without_local_search(self, dummy_data):
        """Test hybrid presolver without local search solvers."""
        features, performance = dummy_data

        presolver = SATzillaHybridPresolver(
            budget=15.0,
            complete_solvers=["algo_0", "algo_2", "algo_4"],
            local_search_solvers=None,
        )
        presolver.fit(features, performance)
        schedule = presolver.predict()

        # Should still work with just complete solvers
        assert isinstance(schedule, list)


class TestPresolverIntegration:
    """Integration tests for presolvers."""

    def test_all_presolvers_same_data(self, dummy_data):
        """Test all presolvers with same data."""
        features, performance = dummy_data
        budget = 15.0

        # Automatic presolver
        auto = SATzillaPresolver(budget=budget)
        auto.fit(features, performance)
        auto_schedule = auto.predict()

        # Manual presolver (using auto's result)
        if auto_schedule:
            manual = SATzillaManualPresolver(schedule=auto_schedule, budget=budget)
            manual.fit(features, performance)
            manual_schedule = manual.predict()
            assert manual_schedule == auto_schedule

        # Hybrid presolver
        hybrid = SATzillaHybridPresolver(
            budget=budget,
            complete_solvers=["algo_0", "algo_2", "algo_4"],
            local_search_solvers=["algo_1", "algo_3"],
        )
        hybrid.fit(features, performance)
        hybrid_schedule = hybrid.predict()

        # All schedules should respect budget
        for schedule in [auto_schedule, hybrid_schedule]:
            total = sum(c for _, c in schedule)
            assert total <= budget

    def test_reproducibility(self, dummy_data):
        """Test that results are reproducible with same random seed."""
        features, performance = dummy_data

        # Fix numpy random seed
        np.random.seed(123)
        presolver1 = SATzillaPresolver(budget=15.0)
        presolver1.fit(features, performance)
        schedule1 = presolver1.predict()

        np.random.seed(123)
        presolver2 = SATzillaPresolver(budget=15.0)
        presolver2.fit(features, performance)
        schedule2 = presolver2.predict()

        assert schedule1 == schedule2
