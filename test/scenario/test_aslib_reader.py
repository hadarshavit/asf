"""Tests for ASlib scenario reader."""

import os
import pytest
import pandas as pd
import arff
import yaml
from asf.scenario.aslib_reader import read_aslib_scenario, evaluate_selector
from asf.selectors.baselines import SingleBestSolver


@pytest.fixture
def aslib_scenario(tmp_path):
    """Create a dummy ASlib scenario."""
    scenario_path = tmp_path / "test_scenario"
    scenario_path.mkdir()

    # 1. description.txt
    description = {
        "scenario_id": "test_scenario",
        "performance_measures": ["runtime"],
        "maximize": False,
        "performance_type": ["runtime"],
        "algorithm_cutoff_time": 100.0,
        "features_cutoff_time": 100.0,
        "features_deterministic": ["f1", "f2"],
        "features_stochastic": [],
        "algorithms": ["algo1", "algo2"],
        "feature_steps": {"step1": {"provides": ["f1", "f2"]}},
        "algorithm_feature_steps": {},
        "metainfo": {},
    }
    with open(scenario_path / "description.txt", "w") as f:
        yaml.dump(description, f)

    # 2. algorithm_runs.arff
    runs_data = {
        "relation": "test_scenario",
        "attributes": [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("algorithm", "STRING"),
            ("runtime", "NUMERIC"),
            ("runstatus", "STRING"),
        ],
        "data": [
            ["i1", 1, "algo1", 10.0, "ok"],
            ["i1", 1, "algo2", 50.0, "ok"],
            ["i2", 1, "algo1", 20.0, "ok"],
            ["i2", 1, "algo2", 5.0, "ok"],
            ["i3", 1, "algo1", 100.0, "ok"],  # Timeout in logic
            ["i3", 1, "algo2", 10.0, "ok"],
        ],
    }
    with open(scenario_path / "algorithm_runs.arff", "w") as f:
        arff.dump(runs_data, f)

    # 3. feature_values.arff
    features_data = {
        "relation": "test_scenario_features",
        "attributes": [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("f1", "NUMERIC"),
            ("f2", "NUMERIC"),
        ],
        "data": [
            ["i1", 1, 0.1, 1.1],
            ["i2", 1, 0.2, 1.2],
            ["i3", 1, 0.3, 1.3],
        ],
    }
    with open(scenario_path / "feature_values.arff", "w") as f:
        arff.dump(features_data, f)

    # 4. cv.arff
    cv_data = {
        "relation": "test_scenario_cv",
        "attributes": [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("fold", "NUMERIC"),
        ],
        "data": [
            ["i1", 1, 1],
            ["i2", 1, 1],
            ["i3", 1, 2],
        ],
    }
    with open(scenario_path / "cv.arff", "w") as f:
        arff.dump(cv_data, f)

    # 5. feature_costs.arff (optional)
    costs_data = {
        "relation": "test_scenario_costs",
        "attributes": [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("step1", "NUMERIC"),
        ],
        "data": [
            ["i1", 1, 0.01],
            ["i2", 1, 0.02],
            ["i3", 1, 0.03],
        ],
    }
    with open(scenario_path / "feature_costs.arff", "w") as f:
        arff.dump(costs_data, f)

    return str(scenario_path)


def test_read_aslib_scenario(aslib_scenario):
    """Test reading basic scenario files."""
    (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    ) = read_aslib_scenario(aslib_scenario)

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (3, 2)
    assert "f1" in features.columns

    assert isinstance(performance, pd.DataFrame)
    assert performance.shape == (3, 2)
    assert "algo1" in performance.columns

    assert isinstance(cv, pd.DataFrame)
    assert cv.shape == (3, 1)  # fold column

    # Check if data alignment is correct by ensuring index matches
    assert all(features.index == performance.index)

    assert maximize is False
    assert budget == 100.0


def test_read_aslib_with_algorithm_features(aslib_scenario):
    """Test reading scenario with algorithm features."""
    path = aslib_scenario

    # Add algorithm features file
    algo_features_data = {
        "relation": "test_scenario_algo_feats",
        "attributes": [
            ("algorithm", "STRING"),
            ("af1", "NUMERIC"),
        ],
        "data": [
            ["algo1", 0.5],
            ["algo2", 0.8],
        ],
    }
    with open(os.path.join(path, "algorithm_feature_values.arff"), "w") as f:
        arff.dump(algo_features_data, f)

    result = read_aslib_scenario(path)
    algorithm_features = result[7]  # 8th element

    assert isinstance(algorithm_features, pd.DataFrame)
    assert algorithm_features.shape == (2, 1)
    assert "af1" in algorithm_features.columns
    assert "algo1" in algorithm_features.index


def test_evaluate_selector_basic(aslib_scenario):
    """Test evaluate_selector function."""

    # 3 instances: i1 (fold 1), i2 (fold 1), i3 (fold 2)
    # If we evaluate on fold 2, train on i1,i2. Test on i3.

    score, selector = evaluate_selector(
        selector_class=SingleBestSolver,
        scenario_path=aslib_scenario,
        fold=2,
    )

    assert isinstance(score, float)
    assert selector is not None


def test_evaluate_selector_per_instance(aslib_scenario):
    """Test evaluate_selector with per-instance return."""

    result = evaluate_selector(
        selector_class=SingleBestSolver,
        scenario_path=aslib_scenario,
        fold=2,
        return_per_instance=True,
    )

    assert len(result) == 3
    score, selector, per_instance = result
    assert isinstance(per_instance, dict)
    # i3 is in fold 2
    assert "i3" in per_instance
