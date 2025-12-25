"""Tests for additional selectors and presolvers to improve coverage."""

import pandas as pd
import pytest


class TestStatic3SPresolver:
    """Tests for Static3S presolver."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("pulp")

    def test_init(self):
        """Test initialization."""
        from asf.presolving.static_3s import Static3S

        presolver = Static3S(budget=30.0)
        assert presolver.budget == 30.0

    def test_fit_with_simple_data(self):
        """Test fit with simple performance data."""
        from asf.presolving.static_3s import Static3S

        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 100.0],
                "algo2": [100.0, 1.0, 2.0],
                "algo3": [50.0, 50.0, 1.0],
            }
        )

        presolver = Static3S(budget=30.0)
        presolver.fit(features=None, performance=performance)

        assert hasattr(presolver, "schedule")

    def test_predict_without_features(self):
        """Test predict returns schedule."""
        from asf.presolving.static_3s import Static3S

        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0, 100.0],
                "algo2": [100.0, 1.0, 2.0],
            }
        )

        presolver = Static3S(budget=30.0)
        presolver.fit(features=None, performance=performance)
        result = presolver.predict()

        assert isinstance(result, list)

    def test_predict_with_features(self):
        """Test predict with features returns dict."""
        from asf.presolving.static_3s import Static3S

        performance = pd.DataFrame(
            {
                "algo1": [1.0, 2.0],
                "algo2": [2.0, 1.0],
            }
        )
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])

        presolver = Static3S(budget=30.0)
        presolver.fit(features=None, performance=performance)
        result = presolver.predict(features=features)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_fit_requires_performance(self):
        """Test that fit requires performance data."""
        from asf.presolving.static_3s import Static3S

        presolver = Static3S(budget=30.0)
        with pytest.raises(ValueError):
            presolver.fit(features=None, performance=None)

    def test_get_preschedule_config(self):
        """Test getting preschedule config."""
        from asf.presolving.static_3s import Static3S

        performance = pd.DataFrame({"algo1": [1.0, 2.0], "algo2": [2.0, 1.0]})

        presolver = Static3S(budget=30.0)
        presolver.fit(features=None, performance=performance)
        config = presolver.get_preschedule_config()

        assert config is not None

    def test_get_configuration(self):
        """Test getting configuration."""
        from asf.presolving.static_3s import Static3S

        performance = pd.DataFrame({"algo1": [1.0, 2.0], "algo2": [2.0, 1.0]})

        presolver = Static3S(budget=30.0)
        presolver.fit(features=None, performance=performance)
        config = presolver.get_configuration()

        assert isinstance(config, dict)
