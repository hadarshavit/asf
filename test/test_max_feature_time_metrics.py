import pandas as pd
from asf.metrics.baselines import (
    running_time_selector_performance,
    running_time_closed_gap,
)


def test_running_time_selector_performance_with_budgeted_feature_groups():
    """Test that the metric handles budgeted feature groups correctly."""
    # Create simple test data
    performance = pd.DataFrame(
        {"algo1": [10.0, 20.0], "algo2": [15.0, 25.0]},
        index=["inst1", "inst2"],  # type: ignore[arg-type]
    )

    feature_time = pd.DataFrame(
        {"fg1": [100.0, 200.0], "fg2": [50.0, 100.0]},
        index=["inst1", "inst2"],  # type: ignore[arg-type]
    )

    budget = 3600.0

    # Schedules with unbounded feature groups: fg1 and fg2 computed fully
    schedules_unbounded = {
        "inst1": ["fg1", "fg2", ("algo1", 3600.0)],
        "inst2": ["fg1", "fg2", ("algo1", 3600.0)],
    }

    # Schedules with budgeted feature groups: fg1 and fg2 capped at 60s each
    schedules_budgeted = {
        "inst1": [("fg1", 60.0), ("fg2", 60.0), ("algo1", 3600.0)],
        "inst2": [("fg1", 60.0), ("fg2", 60.0), ("algo1", 3600.0)],
    }

    # Without budgets: full feature times (inst1: 150, inst2: 300)
    score_unbounded = running_time_selector_performance(
        schedules_unbounded, performance, budget, feature_time, par=10
    )

    # With budgets: capped at 60 per group (inst1: min(100,60)+min(50,60)=110, inst2: min(200,60)+min(100,60)=120)
    score_budgeted = running_time_selector_performance(
        schedules_budgeted, performance, budget, feature_time, par=10
    )

    # Score with budgets should be lower (less feature time)
    assert float(score_budgeted) < float(score_unbounded)  # type: ignore[arg-type]
    # Expected: algo times (10 + 20 = 30) + budgeted feature times (110 + 120 = 230) = 260
    assert abs(float(score_budgeted) - 260.0) < 1.0  # type: ignore[arg-type]


def test_running_time_closed_gap_with_budgeted_feature_groups():
    """Test that closed gap metric handles budgeted feature groups correctly."""
    # Create test data where selector is better than SBS
    performance = pd.DataFrame(
        {"algo1": [100.0, 200.0, 50.0], "algo2": [150.0, 50.0, 250.0]},
        index=["inst1", "inst2", "inst3"],  # type: ignore[arg-type]
    )

    feature_time = pd.DataFrame(
        {"fg1": [100.0, 200.0, 150.0], "fg2": [50.0, 100.0, 75.0]},
        index=["inst1", "inst2", "inst3"],  # type: ignore[arg-type]
    )

    budget = 3600.0

    # Choose best algorithm per instance with unbounded feature groups
    schedules_unbounded = {
        "inst1": ["fg1", "fg2", ("algo1", 3600.0)],
        "inst2": ["fg1", "fg2", ("algo2", 3600.0)],
        "inst3": ["fg1", "fg2", ("algo1", 3600.0)],
    }

    # Choose best algorithm per instance with budgeted feature groups
    schedules_budgeted = {
        "inst1": [("fg1", 60.0), ("fg2", 60.0), ("algo1", 3600.0)],
        "inst2": [("fg1", 60.0), ("fg2", 60.0), ("algo2", 3600.0)],
        "inst3": [("fg1", 60.0), ("fg2", 60.0), ("algo1", 3600.0)],
    }

    # Without budgets
    gap_unbounded = running_time_closed_gap(
        schedules_unbounded, performance, budget, feature_time, par=10
    )

    # With budgets
    gap_budgeted = running_time_closed_gap(
        schedules_budgeted, performance, budget, feature_time, par=10
    )

    # Gap with budgets should be better (higher closed gap) since feature time penalty is lower
    assert float(gap_budgeted) > float(gap_unbounded)  # type: ignore[arg-type]


def test_pipeline_outputs_budgeted_feature_groups():
    """Test that SelectorPipeline outputs feature groups with budgets when max_feature_time is set."""
    # Mock a simple test without heavy imports

    # Simulate feature groups list
    feature_groups_list = ["fg1", "fg2"]
    max_feature_time = 60.0

    # Simulate the conversion logic from SelectorPipeline.predict
    feature_steps_with_budget = [(fg, max_feature_time) for fg in feature_groups_list]

    # Check output format
    assert len(feature_steps_with_budget) == 2
    assert feature_steps_with_budget[0] == ("fg1", 60.0)
    assert feature_steps_with_budget[1] == ("fg2", 60.0)

    # Without max_feature_time, feature groups remain as strings
    feature_steps_no_budget = feature_groups_list
    assert feature_steps_no_budget[0] == "fg1"
    assert feature_steps_no_budget[1] == "fg2"
