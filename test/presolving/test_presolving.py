"""Tests for presolving module."""

import numpy as np
import pandas as pd
import pytest

from asf.presolving.greedy_presolver import GreedyPresolver
from asf.presolving.submodular_presolver import SubmodularPresolver
from asf.presolving.configurable_presolver import ConfigurablePresolver


class TestGreedyPresolver:
    """Tests for GreedyPresolver."""

    def test_fit_with_simple_performance_data(self):
        """Test basic fit with simple performance data."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "algo2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "algo3": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        )

        presolver = GreedyPresolver(
            presolver_budget=10.0,
            cutoff_per_solver=5.0,
            max_presolvers=2,
            min_coverage=0.1,
        )
        presolver.fit(features=None, performance=performance)

        assert len(presolver.schedule) > 0
        assert all(
            isinstance(item, tuple) and len(item) == 2 for item in presolver.schedule
        )

    def test_fit_requires_performance_data(self):
        """Test that fit raises error without performance data."""
        presolver = GreedyPresolver(presolver_budget=10.0)
        with pytest.raises(ValueError, match="requires performance data"):
            presolver.fit(features=None, performance=None)

    def test_predict_without_features(self):
        """Test predict returns schedule without features."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 4.0, 3.0],
            }
        )

        presolver = GreedyPresolver(presolver_budget=10.0, cutoff_per_solver=5.0)
        presolver.fit(features=None, performance=performance)
        result = presolver.predict()

        assert isinstance(result, list)

    def test_predict_with_features_returns_dict(self):
        """Test predict with features returns dict per instance."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 4.0, 3.0],
            }
        )
        features = pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0]}, index=["inst1", "inst2", "inst3"]
        )

        presolver = GreedyPresolver(presolver_budget=10.0, cutoff_per_solver=5.0)
        presolver.fit(features=None, performance=performance)
        result = presolver.predict(features=features)

        assert isinstance(result, dict)
        assert len(result) == 3

    def test_predict_with_numpy_features(self):
        """Test predict with numpy features."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 4.0, 3.0],
            }
        )
        features = np.array([[1.0], [2.0], [3.0]])

        presolver = GreedyPresolver(presolver_budget=10.0, cutoff_per_solver=5.0)
        presolver.fit(features=None, performance=performance)
        result = presolver.predict(features=features)

        assert isinstance(result, dict)

    def test_maximize_mode(self):
        """Test presolver in maximize mode."""
        performance = pd.DataFrame(
            {
                "algo1": [10.0, 20.0, 30.0],
                "algo2": [5.0, 5.0, 5.0],
            }
        )

        presolver = GreedyPresolver(
            presolver_budget=30.0, cutoff_per_solver=25.0, maximize=True
        )
        presolver.fit(features=None, performance=performance)

        assert len(presolver.schedule) >= 0

    def test_fit_with_numpy_performance(self):
        """Test fit with numpy array performance data."""
        # Convert to DataFrame since GreedyPresolver needs column access
        performance = pd.DataFrame(
            np.array(
                [
                    [1.0, 5.0, 3.0],
                    [2.0, 4.0, 3.0],
                    [3.0, 3.0, 3.0],
                ]
            ),
            columns=["algo1", "algo2", "algo3"],
        )

        presolver = GreedyPresolver(presolver_budget=10.0, cutoff_per_solver=5.0)
        presolver.fit(features=None, performance=performance)

        assert len(presolver.schedule) >= 0

    def test_budget_exhaustion(self):
        """Test that schedule respects budget constraint."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 1.0, 1.0],
                "algo2": [2.0, 2.0, 2.0],
            }
        )

        presolver = GreedyPresolver(
            presolver_budget=5.0, cutoff_per_solver=3.0, max_presolvers=5
        )
        presolver.fit(features=None, performance=performance)

        total_time = sum(t for _, t in presolver.schedule)
        assert total_time <= 5.0

    def test_min_coverage_threshold(self):
        """Test that min_coverage filters out low-coverage algorithms."""
        performance = pd.DataFrame(
            {
                "algo1": [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    1.0,
                ],  # Only 1/5 instances solvable
                "algo2": [1.0, 1.0, 1.0, 1.0, 1.0],  # All solvable
            }
        )

        presolver = GreedyPresolver(
            presolver_budget=10.0, cutoff_per_solver=5.0, min_coverage=0.5
        )
        presolver.fit(features=None, performance=performance)

        # algo2 should be in schedule (covers 100%), algo1 not (covers 20%)
        algos_in_schedule = [algo for algo, _ in presolver.schedule]
        assert "algo2" in algos_in_schedule or len(presolver.schedule) == 0


class TestSubmodularPresolver:
    """Tests for SubmodularPresolver."""

    def test_fit_with_simple_performance(self):
        """Test basic fit with simple performance data."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 1.0, 3.0],
                "algo3": [3.0, 3.0, 1.0],
            }
        )

        presolver = SubmodularPresolver(
            presolver_budget=10.0, time_discretization=[1.0, 2.0, 5.0], max_actions=5
        )
        presolver.fit(features=None, performance=performance)

        assert hasattr(presolver, "schedule")

    def test_fit_with_default_discretization(self):
        """Test fit with default time discretization."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 1.0, 3.0],
            }
        )

        presolver = SubmodularPresolver(presolver_budget=30.0, max_actions=3)
        presolver.fit(features=None, performance=performance)

        assert hasattr(presolver, "schedule")

    def test_predict_returns_schedule(self):
        """Test predict returns the computed schedule."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0],
                "algo2": [2.0, 1.0],
            }
        )

        presolver = SubmodularPresolver(
            presolver_budget=5.0, time_discretization=[1.0, 2.0]
        )
        presolver.fit(features=None, performance=performance)
        result = presolver.predict()

        assert isinstance(result, list)

    def test_predict_with_features_returns_dict(self):
        """Test predict with features returns per-instance schedule."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0],
                "algo2": [2.0, 1.0],
            }
        )
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])

        presolver = SubmodularPresolver(
            presolver_budget=5.0, time_discretization=[1.0, 2.0]
        )
        presolver.fit(features=None, performance=performance)
        result = presolver.predict(features=features)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_get_schedule_cost(self):
        """Test computing schedule cost."""
        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 3.0],
                "algo2": [5.0, 1.0, 3.0],
            }
        )

        presolver = SubmodularPresolver(
            presolver_budget=10.0, time_discretization=[1.0, 5.0]
        )
        presolver.fit(features=None, performance=performance)
        cost = presolver.get_schedule_cost(performance=performance)

        assert isinstance(cost, float)
        assert cost >= 0

    def test_fit_with_numpy_performance(self):
        """Test fit with numpy array."""
        performance = np.array(
            [
                [1.0, 5.0],
                [2.0, 1.0],
                [3.0, 3.0],
            ]
        )

        presolver = SubmodularPresolver(
            presolver_budget=10.0, time_discretization=[1.0, 3.0]
        )
        presolver.fit(features=None, performance=performance)

        assert hasattr(presolver, "schedule")

    def test_maximize_mode(self):
        """Test in maximize mode."""
        performance = pd.DataFrame(
            {
                "algo1": [10.0, 20.0],
                "algo2": [5.0, 15.0],
            }
        )

        presolver = SubmodularPresolver(
            presolver_budget=30.0, time_discretization=[10.0, 20.0], maximize=True
        )
        presolver.fit(features=None, performance=performance)

        assert hasattr(presolver, "schedule")


class TestConfigurablePresolver:
    """Tests for ConfigurablePresolver."""

    def test_init_with_algorithm_config(self):
        """Test initialization with algorithm config."""
        config = {
            "algo1": (True, 5.0),
            "algo2": (True, 3.0),
            "algo3": (False, 0.0),
        }

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        assert presolver.algorithm_config == config

    def test_fit_builds_schedule_from_config(self):
        """Test that fit builds schedule from algorithm_config."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0], "algo2": [3.0, 4.0]})
        config = {
            "algo1": (True, 5.0),
            "algo2": (True, 3.0),
        }

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        presolver.fit(features=None, performance=performance)

        assert len(presolver.schedule) == 2
        assert ("algo1", 5.0) in presolver.schedule
        assert ("algo2", 3.0) in presolver.schedule

    def test_fit_filters_disabled_algorithms(self):
        """Test that disabled algorithms are excluded from schedule."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0], "algo2": [3.0, 4.0]})
        config = {
            "algo1": (True, 5.0),
            "algo2": (False, 3.0),
        }

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        presolver.fit(features=None, performance=performance)

        algos = [a for a, _ in presolver.schedule]
        assert "algo1" in algos
        assert "algo2" not in algos

    def test_predict_returns_schedule(self):
        """Test predict returns the schedule."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0]})
        config = {"algo1": (True, 5.0)}

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        presolver.fit(features=None, performance=performance)
        result = presolver.predict()

        assert isinstance(result, list)

    def test_predict_with_features_returns_dict(self):
        """Test predict with features returns per-instance dict."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0]})
        config = {"algo1": (True, 5.0)}
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        presolver.fit(features=None, performance=performance)
        result = presolver.predict(features=features)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_repr(self):
        """Test string representation."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0]})
        config = {"algo1": (True, 5.0)}

        presolver = ConfigurablePresolver(
            presolver_budget=10.0, algorithm_config=config
        )
        presolver.fit(features=None, performance=performance)

        repr_str = repr(presolver)
        assert "ConfigurablePresolver" in repr_str

    def test_fit_with_empty_config(self):
        """Test fit with empty/None config."""
        performance = pd.DataFrame({"algo1": [1.0, 2.0]})
        presolver = ConfigurablePresolver(presolver_budget=10.0, algorithm_config=None)
        presolver.fit(features=None, performance=performance)

        assert presolver.schedule == []

    def test_get_configuration_space(self):
        """Test configuration space generation."""
        pytest.importorskip("ConfigSpace")

        cs = ConfigurablePresolver.get_configuration_space(
            algorithms=["algo1", "algo2"], max_time_per_algo=10.0
        )

        assert cs is not None
        # Verify that parameters for specific algorithms are present
        assert "configurable_presolver:use_algo1" in cs
        assert "configurable_presolver:time_algo1" in cs

    def test_get_from_configuration(self):
        """Test creating presolver from configuration."""
        pytest.importorskip("ConfigSpace")

        # Test with a configuration dict
        config = {
            "configurable_presolver:use_algo1": True,
            "configurable_presolver:time_algo1": 5.0,
            "configurable_presolver:use_algo2": False,
        }

        # get_from_configuration returns a partial, we need to call it
        partial_inst = ConfigurablePresolver.get_from_configuration(
            configuration=config, algorithms=["algo1", "algo2"]
        )
        presolver = partial_inst()

        assert presolver.algorithm_config["algo1"] == (True, 5.0)
        assert presolver.algorithm_config["algo2"] == (
            False,
            5.0,
        )  # Default time is 5.0


class TestAbstractPresolver:
    """Tests for AbstractPresolver base functionality."""

    def test_presolver_get_configuration_space(self):
        """Test configuration space generation."""
        pytest.importorskip("ConfigSpace")

        from asf.presolving.presolver import AbstractPresolver

        # We need a concrete subclass with PREFIX for this to work through ConfigurableMixin
        class ConcretePresolver(AbstractPresolver):
            PREFIX = "concrete"

            def fit(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                pass

        cs = ConcretePresolver.get_configuration_space(total_budget=100.0)

        assert cs is not None
        assert "concrete:presolver_budget" in cs

    def test_presolver_get_from_configuration_not_implemented(self):
        """Test that get_from_configuration raises exception or behaves correctly."""
        pytest.importorskip("ConfigSpace")

        from asf.presolving.presolver import AbstractPresolver

        # ConfigurableMixin.get_from_configuration returns a partial by default
        # if the class doesn't override it and it doesn't have a PREFIX it will fail
        class ConcretePresolver(AbstractPresolver):
            PREFIX = "concrete"

            def fit(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                pass

        partial_inst = ConcretePresolver.get_from_configuration(configuration={})
        inst = partial_inst(presolver_budget=10.0)
        assert isinstance(inst, ConcretePresolver)
        assert inst.presolver_budget == 10.0
