import numpy as np
import pytest
import pandas as pd
from asf.presolving.asap_v2 import ASAPv2


@pytest.fixture
def dummy_data():
    """Create dummy data for testing"""
    features = pd.DataFrame(
        np.random.randn(15, 3), 
        columns=["f1", "f2", "f3"]
    )
    performance = pd.DataFrame(
        np.random.exponential(15, (15, 4)),
        columns=["algo1", "algo2", "algo3", "algo4"]
    )
    return features, performance


def validate_predictions(predictions, n_instances):
    """Basic validation of prediction structure"""
    assert isinstance(predictions, dict)
    assert len(predictions) == n_instances
    
    for schedule in predictions.values():
        assert isinstance(schedule, list), "Schedule should be a list"
        assert len(schedule) > 0, "Schedule should not be empty"
        for alg_name, time_alloc in schedule:
            assert isinstance(alg_name, str), "Algorithm name should be a string"
            assert time_alloc > 0, "Time allocation should be positive"


class TestASAPv2Basic:
    """Essential functionality tests"""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters"""
        asap = ASAPv2(presolver_cutoff=30.0)
        assert asap.budget == 100.0
        assert asap.presolver_cutoff == 30.0
        assert asap.regularization_weight == 0.0
        
        asap_custom = ASAPv2(
            budget=50.0,
            presolver_cutoff=60.0,
            maximize=True,
            regularization_weight=0.5,
            penalty_factor=3.0,
            de_popsize=20,
            seed=123,
            verbosity=1
        )

        error = "Initialization parameter mismatch"
        assert asap_custom.budget == 50.0, error
        assert asap_custom.presolver_cutoff == 60.0, error
        assert asap_custom.maximize, error
        assert asap_custom.regularization_weight == 0.5, error
        assert asap_custom.penalty_factor == 3.0, error
        assert asap_custom.de_popsize == 20, error
        assert asap_custom.de_maxiter == 50.0, error
        assert asap_custom.seed == 123, error
        assert asap_custom.verbosity == 1, error

    def test_fit_and_predict(self, dummy_data):
        """Test basic fit and predict workflow"""
        features, performance = dummy_data
        asap = ASAPv2(presolver_cutoff=30.0, verbosity=0)
        
        asap.fit(features, performance)
        
        assert asap.algorithms == list(performance.columns), "Algorithms do not match performance columns"
        assert len(asap.schedule) > 0, "Schedule should not be empty after fit"
        
        predictions = asap.predict(features)
        validate_predictions(predictions, len(features))

    def test_predict_before_fit_raises_error(self, dummy_data):
        """Test error when predict called before fit"""
        features, _ = dummy_data
        asap = ASAPv2(presolver_cutoff=30.0)
        
        with pytest.raises(ValueError):
            asap.predict(features)

    def test_schedule_ordering(self, dummy_data):
        """Test that schedule is ordered by time allocation"""
        features, performance = dummy_data
        asap = ASAPv2(presolver_cutoff=30.0, verbosity=0)
        asap.fit(features, performance)
        
        times = [time for _, time in asap.schedule]
        assert times == sorted(times), "Schedule should be ordered by time"

    def test_same_schedule_for_all_instances(self, dummy_data):
        """Test ASAP behavior: same schedule for all instances"""
        features, performance = dummy_data
        asap = ASAPv2(presolver_cutoff=30.0, verbosity=0)
        asap.fit(features, performance)
        
        predictions = asap.predict(features)
        schedules = list(predictions.values())
        
        assert all(s == schedules[0] for s in schedules), "All instances should get same schedule"


@pytest.mark.parametrize("presolver_cutoff", [10.0, 30.0, 60.0, 120.0])
def test_schedule_time_allocation(dummy_data, presolver_cutoff):
    """Test that schedule time allocation respects presolver_cutoff"""
    features, performance = dummy_data
    asap = ASAPv2(
        presolver_cutoff=presolver_cutoff,
        budget=5,
        verbosity=0
    )
    
    asap.fit(features, performance)
    
    assert len(asap.schedule) > 0, "Schedule should not be empty"
    
    total_time = sum(time for _, time in asap.schedule)
    assert total_time <= presolver_cutoff + 0.01, f"Total schedule time {total_time} should not exceed presolver_cutoff {presolver_cutoff}"
    
    # Also check that individual algorithm times are reasonable
    for alg_name, time_alloc in asap.schedule:
        assert 0 <= time_alloc <= presolver_cutoff, f"Algorithm {alg_name} time allocation {time_alloc} should be between 0 and presolver_cutoff {presolver_cutoff}"


class TestASAPv2EdgeCases:
    """Test edge cases"""
    
    def test_single_algorithm(self):
        """Test with single algorithm"""
        features = pd.DataFrame(np.random.randn(5, 2), columns=["f1", "f2"])
        performance = pd.DataFrame(
            np.random.exponential(10, (5, 1)), 
            columns=["only_algo"]
        )
        
        asap = ASAPv2(presolver_cutoff=20.0, verbosity=0)
        asap.fit(features, performance)
        
        assert len(asap.schedule) == 1
        assert asap.schedule[0][0] == "only_algo"

    def test_many_algorithms(self):
        """Test with many algorithms"""
        features = pd.DataFrame(np.random.randn(10, 2), columns=["f1", "f2"])
        performance = pd.DataFrame(
            np.random.exponential(15, (10, 8)),
            columns=[f"algo_{i}" for i in range(8)]
        )

        asap = ASAPv2(presolver_cutoff=20.0, budget=5, verbosity=0)
        asap.fit(features, performance)
        
        assert len(asap.algorithms) == 8
        assert len(asap.schedule) <= 8
        assert len(asap.schedule) > 0


@pytest.mark.parametrize("regularization_weight", [0.0, 0.3, 0.7])
def test_regularization_weights(dummy_data, regularization_weight):
    """Test different regularization weights"""
    features, performance = dummy_data
    asap = ASAPv2(
        presolver_cutoff=30.0,
        regularization_weight=regularization_weight,
        budget=5,
        verbosity=0
    )
    
    asap.fit(features, performance)
    
    assert len(asap.schedule) > 0


@pytest.mark.parametrize("seed", [42, 123])
def test_reproducibility(dummy_data, seed):
    """Test reproducibility with same seed"""
    features, performance = dummy_data
    
    asap1 = ASAPv2(presolver_cutoff=30.0, seed=seed, budget=10, verbosity=0)
    asap2 = ASAPv2(presolver_cutoff=30.0, seed=seed, budget=10, verbosity=0)
    
    asap1.fit(features, performance)
    asap2.fit(features, performance)
    
    assert asap1.schedule == asap2.schedule


class TestASAPv2Configuration:
    """Test configuration methods"""
    
    def test_get_preschedule_config(self, dummy_data):
        """Test preschedule config method"""
        features, performance = dummy_data
        asap = ASAPv2(presolver_cutoff=30.0, verbosity=0)
        asap.fit(features, performance)
        
        config = asap.get_preschedule_config()
        assert isinstance(config, dict)
        assert len(config) > 0
        
        for alg, time in config.items():
            assert isinstance(alg, str)
            assert time > 0

    def test_get_configuration(self, dummy_data):
        """Test full configuration method"""
        features, performance = dummy_data
        asap = ASAPv2(
            presolver_cutoff=30.0, 
            regularization_weight=0.1, 
            verbosity=0
        )
        asap.fit(features, performance)
        
        config = asap.get_configuration()
        
        expected_keys = {
            'algorithms', 'budget', 'presolver_cutoff', 
            'preschedule_config', 'regularization_weight', 'penalty_factor'
        }
        assert set(config.keys()) == expected_keys
        assert config['regularization_weight'] == 0.1