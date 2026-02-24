"""Additional tests for abstract selector functionality and prediction modes."""

import numpy as np
import pandas as pd
import pytest

from asf.selectors.baselines import SingleBestSolver


class TestAbstractSelectorPredictionModes:
    """Tests for different prediction modes in AbstractSelector."""

    def test_aslib_mode_returns_dict(self):
        """Test that aslib mode returns a dictionary."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="aslib")
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features)

        assert isinstance(predictions, dict)

    def test_pandas_mode_returns_series(self):
        """Test that pandas mode returns a pandas Series."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="pandas")
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 2

    def test_numpy_mode_returns_onehot(self):
        """Test that numpy mode returns one-hot encoded predictions."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="numpy")
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (2, 2)  # 2 instances, 2 algorithms

    def test_unknown_mode_raises_error(self):
        """Test that unknown prediction mode raises ValueError."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="unknown_mode")
        selector.fit(features=features, performance=performance)

        with pytest.raises(ValueError, match="Unknown prediction_mode"):
            selector.predict(features=features)

    def test_pandas_mode_requires_features(self):
        """Test that pandas mode raises error without features."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="pandas")
        selector.fit(features=features, performance=performance)

        with pytest.raises(ValueError, match="Pandas mode requires features"):
            selector.predict(features=None)

    def test_numpy_mode_requires_features(self):
        """Test that numpy mode raises error without features."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(prediction_mode="numpy")
        selector.fit(features=features, performance=performance)

        with pytest.raises(ValueError, match="Numpy mode requires features"):
            selector.predict(features=None)


class TestAbstractSelectorNumpyInput:
    """Tests for AbstractSelector with numpy array inputs."""

    def test_fit_with_numpy_arrays(self):
        """Test fitting with numpy arrays."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        performance = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])

        selector = SingleBestSolver()
        selector.fit(features=features, performance=performance)

        # Should have automatically created feature names
        assert len(selector.features) == 2
        assert len(selector.algorithms) == 2

    def test_predict_with_numpy_array(self):
        """Test predicting with numpy arrays."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver()
        selector.fit(features=features, performance=performance)

        # Predict with numpy
        test_features = np.array([[3.0]])
        predictions = selector.predict(features=test_features)

        assert isinstance(predictions, dict)


class TestAbstractSelectorLoad:
    """Tests for load method."""

    def test_save_load(self, tmp_path):
        """Test that save and load work correctly."""
        save_path = tmp_path / "selector.joblib"
        selector = SingleBestSolver()
        selector.save(str(save_path))

        loaded = SingleBestSolver.load(str(save_path))
        assert isinstance(loaded, SingleBestSolver)


class TestAbstractSelectorFeatureGroups:
    """Tests for feature groups functionality."""

    def test_feature_groups_added_to_predictions(self):
        """Test that feature groups are prepended to predictions."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 5.0], "algo2": [5.0, 1.0]}, index=["i1", "i2"]
        )

        selector = SingleBestSolver(
            prediction_mode="aslib", feature_groups=["group1", "group2"]
        )
        selector.fit(features=features, performance=performance)
        predictions = selector.predict(features=features)
        assert isinstance(predictions, dict)

        # Predictions should have feature groups prepended
        for _, schedule in predictions.items():
            assert isinstance(schedule, list)
            assert schedule[0] == "group1"
            assert schedule[1] == "group2"
