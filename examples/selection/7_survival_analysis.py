import pandas as pd
import numpy as np
from typing import Sequence, cast
from asf.selectors.survival_analysis import SurvivalAnalysis
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_complex_data(n_instances=100, n_algorithms=6, budget=150.0, seed=42):
    """
    Generates synthetic data to demonstrate the value of survival analysis.

    - Includes a "fast but risky" algorithm with a high timeout probability.
    - Includes a "slow but reliable" algorithm with a low timeout probability.
    - Timeout probability is dependent on instance complexity.
    """
    np.random.seed(seed)

    # Features: Each instance has a "type" that influences which algorithm is best
    types = np.random.choice(
        ["easy", "medium", "hard", "special"], size=n_instances, p=[0.4, 0.3, 0.2, 0.1]
    )
    features = pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 10, n_instances),
            "feature2": np.random.uniform(0, 5, n_instances),
            "feature3": np.random.uniform(0, 1, n_instances),
            "type": types,
        },
        index=pd.Index([f"instance_{i}" for i in range(n_instances)]),
    )

    # Performance: Each algorithm is best for a specific type
    performance = pd.DataFrame(index=features.index)

    # Define timeout probabilities based on problem type
    timeout_probs = {"easy": 0.05, "medium": 0.15, "hard": 0.30, "special": 0.25}

    for algo in range(n_algorithms):
        base = 70 + algo * 15
        perf = base + np.random.normal(0, 10, n_instances)

        # Make each algorithm specialized for a type
        for t in ["easy", "medium", "hard", "special"]:
            mask = features["type"] == t
            if algo == ["easy", "medium", "hard", "special"].index(t):
                perf[mask] -= 30 * algo  # This algorithm is faster for this type
            else:
                perf[mask] += 25  # Slower for other types

        # "Fast but Unreliable" algorithm -> Should not be chosen often
        if algo == 4:
            perf = np.random.uniform(10, 50, n_instances)
            perf_timeouts = np.random.choice(
                [True, False], size=n_instances, p=[0.5, 0.5]
            )
            perf[perf_timeouts] = budget

        # "Slow but Reliable" algorithm -> Should be chosen often
        elif algo == 5:
            perf = np.random.uniform(100, 130, n_instances)
            perf_timeouts = np.random.choice(
                [True, False], size=n_instances, p=[0.05, 0.95]
            )
            perf[perf_timeouts] = budget

        # Apply timeouts based on problem type
        else:
            for t, prob in timeout_probs.items():
                mask = features["type"] == t
                timeouts = np.random.choice(
                    [True, False], size=mask.sum(), p=[prob, 1 - prob]
                )
                perf[mask] = np.where(timeouts, budget, perf[mask])

        performance[f"algo{algo + 1}"] = perf

    # Drop the 'type' column for training/testing
    features = features.drop(columns=["type"])
    return features, performance, types


if __name__ == "__main__":
    BUDGET = 150.0
    features, performance, types = generate_complex_data(budget=BUDGET)

    n_train = int(0.7 * len(features))
    train_idx = features.index[:n_train]
    test_idx = features.index[n_train:]

    train_features = features.loc[train_idx]
    train_performance = performance.loc[train_idx]
    test_features = features.loc[test_idx]
    test_performance = performance.loc[test_idx]
    test_types = types[n_train:]

    # --- 1. Single Best Algorithm Prediction ---
    print("--- Single Best Algorithm Predictions ---")
    selector = SurvivalAnalysis(budget=BUDGET)
    selector.fit(train_features, train_performance)
    predictions = cast(
        dict[str, Sequence[tuple[str, float] | str]], selector.predict(test_features)
    )

    print("Predicted best algorithm for each test instance:")
    for i, instance in enumerate(test_features.index):
        algo, _ = predictions[instance][0]
        true_type = test_types[i]
        print(f"{instance}: {algo} (true type: {true_type})")

    # Wrap predictions with budget for metrics
    budgeted_preds: dict[str, Sequence[tuple[str, float] | str]] = {
        inst: [(algo, BUDGET) for algo, _ in sched]
        for inst, sched in predictions.items()
    }

    # Baselines
    sbs_score = single_best_solver(
        test_performance, maximize=False, budget=BUDGET, par=10.0
    )
    vbs_score = virtual_best_solver(
        test_performance, maximize=False, budget=BUDGET, par=10.0
    )

    # Selector metrics
    sr = compute_solve_rate(budgeted_preds, test_performance, BUDGET)
    par10 = running_time_selector_performance(
        budgeted_preds,
        test_performance,
        budget=BUDGET,
        par=10.0,
        return_per_instance=False,
    )

    print("\n--- Single Best Metrics ---")
    print(f"Budget: {BUDGET}s | Train/Test: {len(train_features)}/{len(test_features)}")
    print(f"Single Best Solver (SBS) PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_score:.2f}")
    print(f"Selector solve-rate: {sr:.2%}")
    print(f"Selector PAR10: {par10:.2f}")

    print("\n" + "=" * 60 + "\n")

    # --- 2. Algorithm Schedule Prediction ---
    print("--- Algorithm Schedule Predictions ---")
    schedule_selector = SurvivalAnalysis(
        budget=BUDGET,
        use_schedule=True,
        max_schedule_length=3,  # Limit schedules to a max of 3 algorithms
        popsize=15,
        maxiter=100,
    )
    schedule_selector.fit(train_features, train_performance)
    schedule_predictions = cast(
        dict[str, Sequence[tuple[str, float] | str]], schedule_selector.predict(test_features)
    )

    print("Predicted algorithm schedules for each test instance:")
    for i, instance in enumerate(test_features.index):
        schedule = schedule_predictions[instance]
        true_type = test_types[i]

        # Format the schedule for printing
        if schedule:
            schedule_str = " -> ".join(
                [f"{algo}({time:.1f}s)" for algo, time in schedule]
            )
        else:
            schedule_str = "No schedule found"

        print(f"{instance} (true type: {true_type}):")
        print(f"  Schedule: {schedule_str}")

    # Schedule selector metrics (schedules already contain time allocations)
    sr_schedule = compute_solve_rate(schedule_predictions, test_performance, BUDGET)
    par10_schedule = running_time_selector_performance(
        schedule_predictions,
        test_performance,
        budget=BUDGET,
        par=10.0,
        return_per_instance=False,
    )

    print("\n--- Schedule Metrics ---")
    print(f"Single Best Solver (SBS) PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_score:.2f}")
    print(f"Schedule Selector solve-rate: {sr_schedule:.2%}")
    print(f"Schedule Selector PAR10: {par10_schedule:.2f}")
