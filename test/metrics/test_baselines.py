import math
import warnings

import pandas as pd
import pytest

from asf.metrics import baselines as m
from asf.preprocessing.feature_group_selector import MissingPrerequisiteGroupError


def test_single_best_solver_minimize_and_maximize():
    perf = pd.DataFrame(
        {
            "A": [1, 3],  # sum = 4
            "B": [2, 5],  # sum = 7
        },
        index=pd.Index(["i1", "i2"]),
    )

    # minimize (default)
    assert m.single_best_solver(perf) == 4

    # maximize
    assert m.single_best_solver(perf, maximize=True) == 7


def test_virtual_best_solver_minimize_and_maximize():
    perf = pd.DataFrame(
        {
            "A": [1, 7],
            "B": [2, 5],
        },
        index=pd.Index(["i1", "i2"]),
    )

    # minimize: pick per-instance minimums: i1=1, i2=5 -> 6
    assert m.virtual_best_solver(perf) == 6

    # maximize: pick per-instance maximums: i1=2, i2=7 -> 9
    assert m.virtual_best_solver(perf, maximize=True) == 9


def test_running_time_selector_performance_solved_and_unsolved():
    perf = pd.DataFrame(
        {
            "algA": [5],
            "algB": [100],
        },
        index=pd.Index(["inst1"]),
    )

    # solved path: allocate more than needed for algA -> solved, include feature time
    schedules = {"inst1": [("algA", 6.0)]}
    feature_time = pd.DataFrame({"feature_time": [1.0]}, index=pd.Index(["inst1"]))
    total = m.running_time_selector_performance(
        schedules, perf, budget=10.0, par=2.0, feature_time=feature_time
    )
    # solved -> time equals true time (5) + feature_time (1) = 6
    assert total == 6.0

    # unsolved path: cannot finish within budget -> budget * par
    schedules_unsolved = {"inst1": [("algB", 2.0)]}
    total_unsolved = m.running_time_selector_performance(
        schedules_unsolved, perf, budget=10.0, par=2.0, feature_time=None
    )
    assert total_unsolved == 20.0


def test_running_time_closed_gap_basic():
    perf = pd.DataFrame(
        {
            "algA": [5, 9],
            "algB": [8, 3],
        },
        index=pd.Index(["i1", "i2"]),
    )
    feature_time = pd.DataFrame(
        {"feature_time": [0.0, 0.0]}, index=pd.Index(["i1", "i2"])
    )

    # schedule solves with exact or more than required budget for each instance
    schedules = {
        "i1": [("algA", 6.0)],  # solved because allocated > needed
        "i2": [("algB", 5.0)],  # solved because allocated > needed
    }

    val = m.running_time_closed_gap(
        schedules=schedules,
        performance=perf,
        budget=20.0,
        feature_time=feature_time,
        par=10.0,
    )

    # sbs = min(sum(col)) = min(5+9=14, 8+3=11) = 11
    # vbs = sum(per-instance min) = min(5,8)+min(9,3) = 5 + 3 = 8
    # s (selector): 5 + 3 = 8 (since both solved exactly)
    # closed gap = (11-8)/(11-8) = 1.0
    assert math.isclose(val, 1.0, rel_tol=1e-9)


def test_precision_regret_behaviors():
    perf = pd.DataFrame(
        {
            "A": [0.1, 0.9],
            "B": [0.7, 0.2],
        },
        index=pd.Index(["x", "y"]),
    )

    # Use performance as precision if precision_data is None
    schedules = {"x": [("B", 1.0)], "y": [("A", 1.0)]}
    # regret is defined in implementation as the selected cell value and then summed
    assert (
        m.precision_regret(schedules, perf) == perf.loc["x", "B"] + perf.loc["y", "A"]
    )

    # With precision_data provided
    precision = pd.DataFrame(
        {
            "A": [0.5, 0.6],
            "B": [0.3, 0.4],
        },
        index=pd.Index(["x", "y"]),
    )
    assert (
        m.precision_regret(schedules, perf, precision_data=precision)
        == precision.loc["x", "B"] + precision.loc["y", "A"]
    )

    # Unknown algorithm and missing instance should be ignored; if all invalid -> warn and return inf
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res = m.precision_regret({"z": [("C", 1.0)]}, perf)
        assert math.isinf(res)
        assert any("No valid schedules" in str(w.message) for w in wlist)


# ============================================================================
# Feature Group Prerequisite Validation Tests
# ============================================================================


class TestSchedulePrerequisiteValidation:
    """Tests for feature group prerequisite validation in schedules."""

    @pytest.fixture
    def feature_groups_with_prereqs(self):
        """Feature groups with prerequisite dependencies."""
        return {
            "Pre": {"provides": ["f1", "f2"]},
            "Basic": {"provides": ["f3", "f4"], "requires": ["Pre"]},
            "CG": {"provides": ["f5", "f6"], "requires": ["Pre"]},
            "Advanced": {"provides": ["f7"], "requires": ["Pre", "Basic"]},
        }

    @pytest.fixture
    def performance(self):
        """Sample performance data."""
        return pd.DataFrame(
            {
                "algo1": [10.0, 20.0],
                "algo2": [15.0, 10.0],
            },
            index=pd.Index(["inst1", "inst2"]),
        )

    @pytest.fixture
    def feature_time(self):
        """Sample feature time data."""
        return pd.DataFrame(
            {
                "Pre": [1.0, 1.0],
                "Basic": [2.0, 2.0],
                "CG": [3.0, 3.0],
            },
            index=pd.Index(["inst1", "inst2"]),
        )

    def test_validate_schedule_valid_prereqs_first(self, feature_groups_with_prereqs):
        """Valid schedule with prerequisites computed first."""
        schedules = {
            "inst1": ["Pre", "Basic", ("algo1", 100)],
            "inst2": ["Pre", "CG", ("algo2", 100)],
        }
        # Should not raise
        m._validate_schedule_prerequisites(schedules, feature_groups_with_prereqs)

    def test_validate_schedule_missing_prereq(self, feature_groups_with_prereqs):
        """Schedule with missing prerequisite should raise error."""
        schedules = {
            "inst1": ["Basic", ("algo1", 100)],  # Basic requires Pre
        }
        with pytest.raises(MissingPrerequisiteGroupError) as exc_info:
            m._validate_schedule_prerequisites(schedules, feature_groups_with_prereqs)

        assert "Basic" in str(exc_info.value)
        assert "Pre" in str(exc_info.value)
        assert "inst1" in str(exc_info.value)

    def test_validate_schedule_prereq_after_dependent(
        self, feature_groups_with_prereqs
    ):
        """Schedule with prerequisite appearing after dependent should raise."""
        schedules = {
            "inst1": ["Basic", "Pre", ("algo1", 100)],  # Pre must come before Basic
        }
        with pytest.raises(MissingPrerequisiteGroupError) as exc_info:
            m._validate_schedule_prerequisites(schedules, feature_groups_with_prereqs)

        assert "Basic" in str(exc_info.value)
        assert "Pre" in str(exc_info.value)

    def test_validate_schedule_no_feature_groups(self, feature_groups_with_prereqs):
        """Schedule without feature groups should pass validation."""
        schedules = {
            "inst1": [("algo1", 100)],
            "inst2": [("algo2", 50)],
        }
        # Should not raise
        m._validate_schedule_prerequisites(schedules, feature_groups_with_prereqs)

    def test_validate_schedule_chained_prereqs(self, feature_groups_with_prereqs):
        """Schedule with chained prerequisites should validate correctly."""
        # Advanced requires both Pre and Basic

        # Valid: Pre -> Basic -> Advanced
        valid_schedules = {
            "inst1": ["Pre", "Basic", "Advanced", ("algo1", 100)],
        }
        m._validate_schedule_prerequisites(valid_schedules, feature_groups_with_prereqs)

        # Invalid: Pre -> Advanced (missing Basic)
        invalid_schedules = {
            "inst1": ["Pre", "Advanced", ("algo1", 100)],
        }
        with pytest.raises(MissingPrerequisiteGroupError):
            m._validate_schedule_prerequisites(
                invalid_schedules, feature_groups_with_prereqs
            )

    def test_validate_schedule_unknown_group(self, feature_groups_with_prereqs):
        """Unknown feature groups in schedule should be handled gracefully."""
        schedules = {
            "inst1": ["UnknownGroup", "Pre", ("algo1", 100)],
        }
        # Should not raise - unknown groups are ignored
        m._validate_schedule_prerequisites(schedules, feature_groups_with_prereqs)


class TestClosedGapPrerequisiteValidation:
    """Tests for prerequisite validation in running_time_closed_gap."""

    @pytest.fixture
    def feature_groups_with_prereqs(self):
        return {
            "Pre": {"provides": ["f1"]},
            "Basic": {"provides": ["f2"], "requires": ["Pre"]},
        }

    @pytest.fixture
    def performance(self):
        return pd.DataFrame(
            {
                "algo1": [10.0, 20.0],
                "algo2": [15.0, 10.0],
            },
            index=pd.Index(["inst1", "inst2"]),
        )

    @pytest.fixture
    def feature_time(self):
        return pd.DataFrame(
            {
                "Pre": [1.0, 1.0],
                "Basic": [2.0, 2.0],
            },
            index=pd.Index(["inst1", "inst2"]),
        )

    def test_closed_gap_valid_schedule(
        self, feature_groups_with_prereqs, performance, feature_time
    ):
        """closed_gap with valid schedule should compute result."""
        schedules = {
            "inst1": ["Pre", "Basic", ("algo1", 100)],
            "inst2": ["Pre", ("algo2", 100)],
        }
        result = m.running_time_closed_gap(
            schedules,
            performance,
            budget=100,
            feature_time=feature_time,
            feature_groups=feature_groups_with_prereqs,
        )
        assert isinstance(result, float)

    def test_closed_gap_invalid_schedule(
        self, feature_groups_with_prereqs, performance, feature_time
    ):
        """closed_gap with invalid schedule should raise error."""
        schedules = {
            "inst1": ["Basic", ("algo1", 100)],  # Missing Pre
            "inst2": ["Pre", ("algo2", 100)],
        }
        with pytest.raises(MissingPrerequisiteGroupError):
            m.running_time_closed_gap(
                schedules,
                performance,
                budget=100,
                feature_time=feature_time,
                feature_groups=feature_groups_with_prereqs,
            )

    def test_closed_gap_no_feature_groups_no_validation(
        self, performance, feature_time
    ):
        """closed_gap without feature_groups should skip validation."""
        # Invalid schedule that would fail validation
        schedules = {
            "inst1": ["Basic", ("algo1", 100)],
            "inst2": [("algo2", 100)],
        }
        # Should not raise when feature_groups is None
        result = m.running_time_closed_gap(
            schedules,
            performance,
            budget=100,
            feature_time=feature_time,
            feature_groups=None,
        )
        assert isinstance(result, float)

    def test_closed_gap_backward_compatible(self, performance, feature_time):
        """closed_gap should work without feature_groups parameter (backward compat)."""
        schedules = {
            "inst1": [("algo1", 100)],
            "inst2": [("algo2", 100)],
        }
        # Should work without feature_groups
        result = m.running_time_closed_gap(
            schedules, performance, budget=100, feature_time=feature_time
        )
        assert isinstance(result, float)
