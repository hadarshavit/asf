import numpy as np
import pandas as pd
import pytest

from asf.presolving.static_3s import Static3S

try:
    import pulp  # noqa: F401

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(0)
    features = pd.DataFrame(rng.randn(20, 3), columns=pd.Index(["f1", "f2", "f3"]))
    performance = pd.DataFrame(
        rng.exponential(15, (20, 4)),
        columns=pd.Index(["algo1", "algo2", "algo3", "algo4"]),
    )
    return features, performance


def _validate_predictions(preds, n):
    assert isinstance(preds, dict) and len(preds) == n
    for sched in preds.values():
        assert isinstance(sched, list)
        for entry in sched:
            assert isinstance(entry, tuple) and len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], (int, float, np.floating))


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp is not installed")
def test_static3s_basic_flow_and_predict(dummy_data):
    X, Y = dummy_data
    s = Static3S(runcount_limit=5, presolver_budget=30.0, max_candidates_per_solver=8)
    s.fit(X, Y)
    assert isinstance(s.schedule, list)
    s.fit(X, Y)
    assert isinstance(s.schedule, list)
    preds = s.predict()
    assert preds == s.schedule


def test_predict_before_fit_raises(dummy_data):
    X, _ = dummy_data
    s = Static3S(presolver_budget=30.0)
    with pytest.raises(ValueError, match="Static3S has not been fitted yet"):
        s.predict()


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp is not installed")
def test_schedule_time_and_ordering(dummy_data):
    X, Y = dummy_data
    budget = 40.0
    s = Static3S(runcount_limit=5, presolver_budget=budget, max_candidates_per_solver=8)
    s.fit(X, Y)
    assert s.schedule is not None
    assert len(s.schedule) > 0
    times = [t for _, t in s.schedule]
    assert times == sorted(times)
    assert sum(times) == pytest.approx(budget, rel=1e-6)


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp is not installed")
def test_configuration_and_algorithms(dummy_data):
    X, Y = dummy_data
    s = Static3S(presolver_budget=30.0)
    s.fit(X, Y)
    cfg = s.get_configuration()
    assert "algorithms" in cfg and cfg["algorithms"] == list(Y.columns)
    assert "budget" in cfg and cfg["budget"] == pytest.approx(30.0)
    assert "preschedule_config" in cfg and isinstance(cfg["preschedule_config"], dict)
