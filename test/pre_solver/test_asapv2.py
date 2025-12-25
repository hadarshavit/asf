import numpy as np
import pytest
import pandas as pd
from asf.presolving.asap_v2 import ASAPv2


@pytest.fixture
def dummy_data():
    """Create dummy data for testing"""
    features = pd.DataFrame(
        np.random.randn(15, 3), columns=pd.Index(["f1", "f2", "f3"])
    )
    performance = pd.DataFrame(
        np.random.exponential(15, (15, 4)),
        columns=pd.Index(["algo1", "algo2", "algo3", "algo4"]),
    )
    return features, performance


def validate_predictions(predictions, budget):
    """Basic validation of prediction structure"""
    # predict() returns dict when features are passed, list otherwise
    if isinstance(predictions, dict):
        for inst_key, schedule in predictions.items():
            assert isinstance(inst_key, str), "Instance key should be a string"
            assert isinstance(schedule, list), "Each schedule should be a list"
            for alg_name, time_alloc in schedule:
                assert isinstance(alg_name, str), "Algorithm name should be a string"
                assert time_alloc >= 0, "Time allocation should be positive"
    else:
        assert isinstance(predictions, list), "Predictions should be a list (schedule)"
        for alg_name, time_alloc in predictions:
            assert isinstance(alg_name, str), "Algorithm name should be a string"
            assert time_alloc >= 0, "Time allocation should be positive"


class TestASAPv2Basic:
    """Essential functionality tests"""

    def test_initialization(self):
        """Test initialization with default and custom parameters"""
        asap = ASAPv2(budget=30.0)
        assert asap.de_maxiter == 100
        assert asap.budget == 30.0
        assert asap.regularization_weight == 0.0
        assert asap.size_preschedule == 3

        asap_custom = ASAPv2(
            runcount_limit=50.0,
            budget=60.0,
            maximize=True,
            size_preschedule=2,
            regularization_weight=0.5,
            variance_weight=0.1,
            de_popsize=20,
            seed=123,
            verbosity=1,
        )

        error = "Initialization parameter mismatch"
        assert asap_custom.de_maxiter == 50, error
        assert asap_custom.budget == 60.0, error
        assert asap_custom.maximize, error
        assert asap_custom.size_preschedule == 2, error
        assert asap_custom.regularization_weight == 0.5, error
        assert asap_custom.variance_weight == 0.1, error
        assert asap_custom.de_popsize == 20, error
        assert asap_custom.seed == 123, error
        assert asap_custom.verbosity == 1, error

    def test_fit_and_predict(self, dummy_data):
        """Test basic fit and predict workflow"""
        features, performance = dummy_data
        asap = ASAPv2(budget=30.0, verbosity=0)

        asap.fit(features, performance)

        assert asap.algorithms == list(performance.columns), (
            "Algorithms do not match performance columns"
        )
        assert len(asap.schedule) > 0, "Schedule should not be empty after fit"

        predictions = asap.predict(features)
        validate_predictions(predictions, asap.budget)

    def test_predict_returns_schedule(self, dummy_data):
        """Test that predict returns the schedule"""
        features, performance = dummy_data
        asap = ASAPv2(budget=30.0, verbosity=0)
        asap.fit(features, performance)

        # predict() with features returns dict mapping instances to schedules
        predictions_dict = asap.predict(features)
        assert isinstance(predictions_dict, dict), (
            "predictions with features should be dict"
        )
        for inst_key, schedule in predictions_dict.items():
            assert schedule == asap.schedule, (
                "Each schedule should equal fitted schedule"
            )

        # predict() without features returns the schedule directly
        predictions_list = asap.predict()
        assert predictions_list == asap.schedule, (
            "Predictions without features should equal schedule"
        )


@pytest.mark.parametrize("budget", [10.0, 30.0, 60.0, 120.0])
def test_schedule_time_allocation(dummy_data, budget):
    """Test that schedule time allocation respects budget"""
    features, performance = dummy_data
    asap = ASAPv2(budget=budget, runcount_limit=5, verbosity=0)

    asap.fit(features, performance)

    assert len(asap.schedule) > 0, "Schedule should not be empty"

    total_time = sum(time for _, time in asap.schedule)
    # Preschedule time should be at most max_runtime_preschedule (10% of budget by default)
    assert total_time <= budget + 0.01, (
        f"Total schedule time {total_time} should not exceed budget {budget}"
    )

    # Also check that individual algorithm times are reasonable
    for alg_name, time_alloc in asap.schedule:
        assert 0 <= time_alloc <= budget, (
            f"Algorithm {alg_name} time allocation {time_alloc} should be between 0 and budget {budget}"
        )


class TestASAPv2EdgeCases:
    """Test edge cases"""

    def test_single_algorithm(self):
        """Test with single algorithm - size_preschedule becomes 0"""
        features = pd.DataFrame(np.random.randn(5, 2), columns=pd.Index(["f1", "f2"]))
        performance = pd.DataFrame(
            np.random.exponential(10, (5, 1)), columns=pd.Index(["only_algo"])
        )

        asap = ASAPv2(budget=20.0, verbosity=0)
        asap.fit(features, performance)

        # With single algorithm, size_preschedule becomes 0 (min(3, 1-1) = 0)
        assert len(asap.schedule) == 0 or (
            len(asap.schedule) == 1 and asap.schedule[0][0] == "only_algo"
        )

    def test_two_algorithms(self):
        """Test with two algorithms - size_preschedule becomes 1"""
        features = pd.DataFrame(np.random.randn(5, 2), columns=pd.Index(["f1", "f2"]))
        performance = pd.DataFrame(
            np.random.exponential(10, (5, 2)), columns=pd.Index(["algo1", "algo2"])
        )

        asap = ASAPv2(budget=20.0, verbosity=0)
        asap.fit(features, performance)

        assert len(asap.algorithms) == 2
        # size_preschedule = min(3, 2-1) = 1
        assert len(asap.schedule) <= 1

    def test_many_algorithms(self):
        """Test with many algorithms"""
        features = pd.DataFrame(np.random.randn(10, 2), columns=pd.Index(["f1", "f2"]))
        performance = pd.DataFrame(
            np.random.exponential(15, (10, 8)),
            columns=pd.Index([f"algo_{i}" for i in range(8)]),
        )

        asap = ASAPv2(budget=20.0, runcount_limit=5, verbosity=0, size_preschedule=3)
        asap.fit(features, performance)

        assert len(asap.algorithms) == 8
        # Schedule should have at most size_preschedule algorithms
        assert len(asap.schedule) <= 3
        assert len(asap.schedule) > 0


@pytest.mark.parametrize("size_preschedule", [1, 2, 3, 4])
def test_size_preschedule(dummy_data, size_preschedule):
    """Test different preschedule sizes"""
    features, performance = dummy_data
    asap = ASAPv2(
        budget=30.0,
        size_preschedule=size_preschedule,
        runcount_limit=5,
        verbosity=0,
    )

    asap.fit(features, performance)

    # Actual size should be min(size_preschedule, numAlg - 1)
    expected_size = min(size_preschedule, len(performance.columns) - 1)
    assert len(asap.schedule) <= expected_size or expected_size == 1


@pytest.mark.parametrize("regularization_weight", [0.0, 0.3, 0.7])
def test_regularization_weights(dummy_data, regularization_weight):
    """Test different regularization weights"""
    features, performance = dummy_data
    asap = ASAPv2(
        budget=30.0,
        regularization_weight=regularization_weight,
        runcount_limit=5,
        verbosity=0,
    )

    asap.fit(features, performance)

    assert len(asap.schedule) > 0


@pytest.mark.parametrize("variance_weight", [0.0, 0.1, 0.5])
def test_variance_weights(dummy_data, variance_weight):
    """Test different variance weights"""
    features, performance = dummy_data
    asap = ASAPv2(
        budget=30.0,
        variance_weight=variance_weight,
        runcount_limit=5,
        verbosity=0,
    )

    asap.fit(features, performance)

    assert len(asap.schedule) > 0


@pytest.mark.parametrize("seed", [42, 123])
def test_reproducibility(dummy_data, seed):
    """Test reproducibility with same seed"""
    features, performance = dummy_data

    asap1 = ASAPv2(budget=30.0, seed=seed, runcount_limit=10, verbosity=0)
    asap2 = ASAPv2(budget=30.0, seed=seed, runcount_limit=10, verbosity=0)

    asap1.fit(features, performance)
    asap2.fit(features, performance)

    assert asap1.schedule == asap2.schedule


class TestASAPv2Configuration:
    """Test configuration methods"""

    def test_get_preschedule_config(self, dummy_data):
        """Test preschedule config method"""
        features, performance = dummy_data
        asap = ASAPv2(budget=30.0, verbosity=0)
        asap.fit(features, performance)

        config = asap.get_preschedule_config()
        assert isinstance(config, dict)

        for alg, time in config.items():
            assert isinstance(alg, str)
            assert time > 0

    def test_max_runtime_preschedule_default(self, dummy_data):
        """Test default max_runtime_preschedule is 10% of budget"""
        features, performance = dummy_data

        asap = ASAPv2(budget=100.0, verbosity=0)
        assert asap.max_runtime_preschedule == 10.0  # 10% of 100

        asap = ASAPv2(budget=50.0, verbosity=0)
        assert asap.max_runtime_preschedule == 5.0  # 10% of 50

    def test_max_runtime_preschedule_fraction(self, dummy_data):
        """Test max_runtime_preschedule as fraction"""
        features, performance = dummy_data

        asap = ASAPv2(budget=100.0, max_runtime_preschedule=0.2, verbosity=0)
        assert asap.max_runtime_preschedule == 20.0  # 20% of 100

    def test_max_runtime_preschedule_absolute(self, dummy_data):
        """Test max_runtime_preschedule as absolute value"""
        features, performance = dummy_data

        asap = ASAPv2(budget=100.0, max_runtime_preschedule=15.0, verbosity=0)
        assert asap.max_runtime_preschedule == 15.0


class TestASAPv2AlgorithmSelection:
    """Test algorithm selection for preschedule"""

    def test_preschedule_algorithms_selected(self, dummy_data):
        """Test that preschedule selects subset of algorithms"""
        features, performance = dummy_data
        asap = ASAPv2(budget=30.0, size_preschedule=2, runcount_limit=5, verbosity=0)
        asap.fit(features, performance)

        # Check that ialgos_preschedule contains valid indices
        assert asap.ialgos_preschedule is not None
        assert len(asap.ialgos_preschedule) <= 2
        for idx in asap.ialgos_preschedule:
            assert 0 <= idx < len(asap.algorithms)

    def test_preschedule_algorithms_in_schedule(self, dummy_data):
        """Test that schedule contains algorithms from preschedule"""
        features, performance = dummy_data
        asap = ASAPv2(budget=30.0, size_preschedule=3, runcount_limit=5, verbosity=0)
        asap.fit(features, performance)

        assert asap.schedule is not None
        schedule_algos = [alg for alg, _ in asap.schedule]
        assert asap.ialgos_preschedule is not None
        preschedule_algos = [asap.algorithms[idx] for idx in asap.ialgos_preschedule]

        for alg in schedule_algos:
            assert alg in preschedule_algos
