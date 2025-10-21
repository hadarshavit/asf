import math
import warnings

import pandas as pd

from asf.metrics import baselines as m


def test_single_best_solver_minimize_and_maximize():
    perf = pd.DataFrame(
        {
            "A": [1, 3],  # sum = 4
            "B": [2, 5],  # sum = 7
        },
        index=["i1", "i2"],
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
        index=["i1", "i2"],
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
        index=["inst1"],
    )

    # solved path: allocate more than needed for algA -> solved, include feature time
    schedules = {"inst1": [("algA", 6.0)]}
    feature_time = pd.DataFrame({"feature_time": [1.0]}, index=["inst1"])
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
        index=["i1", "i2"],
    )
    feature_time = pd.DataFrame({"feature_time": [0.0, 0.0]}, index=["i1", "i2"])

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
        index=["x", "y"],
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
        index=["x", "y"],
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
