"""Tests for PAR10 metric utilities."""

import numpy as np
import pandas as pd

from asf.metrics.par10 import apply_par, apply_par10


class TestApplyPar:
    """Tests for apply_par function."""

    def test_apply_par_dataframe(self):
        """Test PAR transformation on DataFrame."""
        performance = pd.DataFrame(
            {
                "algo1": [100.0, 1201.0, 500.0],
                "algo2": [200.0, 200.0, 1201.0],
            }
        )
        budget = 1200.0

        result = apply_par(performance, budget, par_factor=10.0)

        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "algo1"] == 100.0
        assert result.loc[1, "algo1"] == 12000.0  # Penalized
        assert result.loc[0, "algo2"] == 200.0
        assert result.loc[2, "algo2"] == 12000.0  # Penalized

    def test_apply_par_numpy(self):
        """Test PAR transformation on numpy array."""
        performance = np.array(
            [
                [100.0, 200.0],
                [1201.0, 200.0],
                [500.0, 1201.0],
            ]
        )
        budget = 1200.0

        result = apply_par(performance, budget, par_factor=10.0)

        assert isinstance(result, np.ndarray)
        assert result[0, 0] == 100.0
        assert result[1, 0] == 12000.0  # Penalized
        assert result[2, 1] == 12000.0  # Penalized

    def test_apply_par_no_timeouts(self):
        """Test when there are no timeouts."""
        performance = pd.DataFrame(
            {
                "algo1": [100.0, 200.0, 500.0],
                "algo2": [150.0, 300.0, 600.0],
            }
        )
        budget = 1200.0

        result = apply_par(performance, budget, par_factor=10.0)

        pd.testing.assert_frame_equal(result, performance)

    def test_apply_par_all_timeouts(self):
        """Test when all values are timeouts."""
        performance = np.array([[1300.0, 1400.0], [1500.0, 1600.0]])
        budget = 1200.0

        result = apply_par(performance, budget, par_factor=10.0)

        expected = np.array([[12000.0, 12000.0], [12000.0, 12000.0]])
        np.testing.assert_array_equal(result, expected)

    def test_apply_par_different_factors(self):
        """Test with different PAR factors."""
        performance = np.array([[100.0, 1201.0]])
        budget = 1200.0

        result_par2 = apply_par(performance, budget, par_factor=2.0)
        result_par10 = apply_par(performance, budget, par_factor=10.0)

        assert result_par2[0, 1] == 2400.0
        assert result_par10[0, 1] == 12000.0

    def test_apply_par_boundary_values(self):
        """Test values exactly at the boundary."""
        performance = np.array([[1200.0, 1200.001]])
        budget = 1200.0

        result = apply_par(performance, budget, par_factor=10.0)

        assert result[0, 0] == 1200.0  # Exactly at budget, not penalized
        assert result[0, 1] == 12000.0  # Slightly over, penalized


class TestApplyPar10:
    """Tests for apply_par10 convenience function."""

    def test_apply_par10_dataframe(self):
        """Test PAR10 on DataFrame."""
        performance = pd.DataFrame({"algo1": [100.0, 1201.0]})
        budget = 1200.0

        result = apply_par10(performance, budget)

        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "algo1"] == 100.0
        assert result.loc[1, "algo1"] == 12000.0

    def test_apply_par10_numpy(self):
        """Test PAR10 on numpy array."""
        performance = np.array([[100.0, 1201.0]])
        budget = 1200.0

        result = apply_par10(performance, budget)

        assert isinstance(result, np.ndarray)
        assert result[0, 0] == 100.0
        assert result[0, 1] == 12000.0
