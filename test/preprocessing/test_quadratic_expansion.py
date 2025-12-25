"""Tests for quadratic expansion / schmee_hahn_impute."""

import numpy as np

from asf.preprocessing.quadratic_expansion import schmee_hahn_impute


class MockEPM:
    """Mock EPM model for testing."""

    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        """Fit the mock model."""
        self.fitted = True
        self.X_ = X
        self.y_ = y
        # Store mean for predictions
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        """Return predictions."""
        # Return the mean value for all predictions
        return np.full(len(X), self.mean_)


class TestSchmeeHahnImpute:
    """Tests for schmee_hahn_impute function."""

    def test_basic_imputation(self):
        """Test basic imputation with some censored values."""
        # Create some test data
        y_raw = np.array([10.0, 20.0, 300.0, 50.0, 300.0])  # 300 = timeout
        X_exp = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM, y_raw=y_raw, X_exp=X_exp, budget=budget
        )

        assert y_imputed is not None
        assert len(y_imputed) == len(y_raw)
        # Non-timeout values should be preserved
        assert y_imputed[0] == 10.0
        assert y_imputed[1] == 20.0
        assert y_imputed[3] == 50.0
        # Model should be fitted
        assert model is not None

    def test_no_censored_values(self):
        """Test when there are no censored values."""
        y_raw = np.array([10.0, 20.0, 30.0, 50.0])
        X_exp = np.array([[1.0], [2.0], [3.0], [4.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM, y_raw=y_raw, X_exp=X_exp, budget=budget
        )

        # With no censored values, output should match input
        np.testing.assert_array_equal(y_imputed, y_raw)

    def test_all_censored_values(self):
        """Test when all values are censored."""
        y_raw = np.array([300.0, 300.0, 300.0])
        X_exp = np.array([[1.0], [2.0], [3.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM, y_raw=y_raw, X_exp=X_exp, budget=budget
        )

        assert y_imputed is not None
        assert len(y_imputed) == len(y_raw)

    def test_all_nan_returns_nan_and_none(self):
        """Test when all values are NaN."""
        y_raw = np.array([np.nan, np.nan, np.nan])
        X_exp = np.array([[1.0], [2.0], [3.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM, y_raw=y_raw, X_exp=X_exp, budget=budget
        )

        assert np.all(np.isnan(y_imputed))
        assert model is None

    def test_convergence_parameters(self):
        """Test with different EM parameters."""
        y_raw = np.array([10.0, 20.0, 300.0, 50.0])
        X_exp = np.array([[1.0], [2.0], [3.0], [4.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM,
            y_raw=y_raw,
            X_exp=X_exp,
            budget=budget,
            em_max_iter=5,
            em_tol=0.01,
            em_min_sigma=1e-4,
        )

        assert y_imputed is not None
        assert model is not None

    def test_mixed_nan_and_censored(self):
        """Test with mix of NaN and censored values."""
        y_raw = np.array([10.0, np.nan, 300.0, 50.0, np.nan])
        X_exp = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        budget = 300.0

        y_imputed, model = schmee_hahn_impute(
            base_epm=MockEPM, y_raw=y_raw, X_exp=X_exp, budget=budget
        )

        assert y_imputed is not None
        # NaN values should remain NaN
        assert np.isnan(y_imputed[1])
        assert np.isnan(y_imputed[4])
        # Non-NaN non-timeout values should be preserved
        assert y_imputed[0] == 10.0
        assert y_imputed[3] == 50.0
