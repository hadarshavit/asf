import pandas as pd
import numpy as np
from typing import cast

from asf.selectors.sunny import SUNNY
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_simple_data(n_instances=100, seed=0):
    """Generate more challenging data with timeouts and varying algorithm behavior."""
    np.random.seed(seed)

    # Create features with 3 dimensions
    features = pd.DataFrame(
        np.random.uniform(0, 10, size=(n_instances, 3)),
        columns=pd.Index(["size", "density", "complexity"]),
        index=pd.Index([f"inst_{i}" for i in range(n_instances)]),
    )

    # Create 6 algorithms with different characteristics
    performance = pd.DataFrame(index=features.index)

    for i, instance in enumerate(features.index):
        size = features.loc[instance, "size"]
        density = features.loc[instance, "density"]
        complexity = features.loc[instance, "complexity"]

        # algo1: Fast on small instances, but 40% timeout on large ones
        if size > 7 and np.random.random() < 0.4:
            performance.loc[instance, "algo1"] = 250  # timeout
        else:
            performance.loc[instance, "algo1"] = max(
                5, 30 + 3 * size + np.random.normal(0, 5)
            )

        # algo2: Slow but very reliable
        performance.loc[instance, "algo2"] = max(
            20, 80 + 5 * density + 2 * complexity + np.random.normal(0, 8)
        )

        # algo3: Good on high density, 30% timeout on low density
        if density < 3 and np.random.random() < 0.3:
            performance.loc[instance, "algo3"] = 250  # timeout
        else:
            performance.loc[instance, "algo3"] = max(
                10, 100 - 8 * density + 3 * size + np.random.normal(0, 6)
            )

        # algo4: Fast but 50% timeout on complex instances
        if complexity > 7 and np.random.random() < 0.5:
            performance.loc[instance, "algo4"] = 250  # timeout
        else:
            performance.loc[instance, "algo4"] = max(
                8, 25 + 2 * complexity + np.random.normal(0, 7)
            )

        # algo5: Mediocre but reliable
        performance.loc[instance, "algo5"] = max(
            15, 50 + 3 * size + 2 * density + np.random.normal(0, 10)
        )

        # algo6: Excellent on low complexity, 35% timeout on high complexity
        if complexity > 6 and np.random.random() < 0.35:
            performance.loc[instance, "algo6"] = 250  # timeout
        else:
            performance.loc[instance, "algo6"] = max(
                10, 40 - 5 * complexity + 4 * density + np.random.normal(0, 6)
            )

    return features, performance


def print_sunny_schedules(predictions, true_performance, budget, n=10):
    print("\nSample SUNNY schedules:")
    for instance in list(true_performance.index)[:n]:
        schedule = predictions[instance]
        row = true_performance.loc[instance]
        solved_algo = None
        time_used = 0
        solved = False
        for algo, t in schedule:
            runtime = row[algo]
            time_used += t
            if runtime <= budget and runtime <= time_used:
                solved_algo = algo
                solved = True
                break
        if solved:
            emoji = "✅"
        else:
            emoji = "❌"
        schedule_str = " -> ".join([f"{a}({int(t)}s)" for a, t in schedule])
        print(
            f"{instance}: {schedule_str} | "
            f"Solved by: {solved_algo} {emoji} | "
            f"#algos: {len(schedule)}"
        )


if __name__ == "__main__":
    features, performance = generate_simple_data(n_instances=100, seed=42)
    budget = 200

    # Split into train/test
    n_train = int(0.7 * len(features))
    train_features = features.iloc[:n_train]
    train_performance = performance.iloc[:n_train]
    test_features = features.iloc[n_train:]
    test_performance = performance.iloc[n_train:]

    print("=" * 70)
    print("SUNNY Selector Comparison")
    print("=" * 70)
    print(f"Training set: {len(train_features)} instances")
    print(f"Test set: {len(test_features)} instances")
    print(f"Budget: {budget}s")
    print(f"Algorithms: {list(performance.columns)}")

    # Baselines
    sbs_score = single_best_solver(
        test_performance, maximize=False, budget=budget, par=10.0
    )
    vbs_score = virtual_best_solver(
        test_performance, maximize=False, budget=budget, par=10.0
    )

    print(f"\nSingle Best Solver PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10: {vbs_score:.2f}")

    # 1. Basic SUNNY (no tuning)
    print("\n" + "=" * 70)
    print("1. Basic SUNNY (fixed k=5, no tuning)")
    print("=" * 70)
    selector = SUNNY(k=5, use_v2=False, budget=budget)
    selector.fit(train_features, train_performance)
    predictions = cast(
        dict[str, list[tuple[str, float]]], selector.predict(test_features)
    )
    sr = compute_solve_rate(predictions, test_performance, budget)
    par10 = running_time_selector_performance(
        predictions,
        test_performance,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"k = {selector.k}")
    print("Algorithm limit: None")
    print(f"Solve rate: {sr:.1%}  PAR10: {par10:.2f}")
    print_sunny_schedules(predictions, test_performance, budget, n=8)

    # 2. SUNNY-AS2 (k-tuning)
    print("\n" + "=" * 70)
    print("2. SUNNY-AS2 (k-tuning enabled)")
    print("=" * 70)
    selector_v2 = SUNNY(k=5, use_v2=True, budget=budget, k_candidates=[3, 5, 7, 10, 15])
    selector_v2.fit(train_features, train_performance)
    predictions_v2 = cast(
        dict[str, list[tuple[str, float]]], selector_v2.predict(test_features)
    )
    sr_v2 = compute_solve_rate(predictions_v2, test_performance, budget)
    par10_v2 = running_time_selector_performance(
        predictions_v2,
        test_performance,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"Tuned k = {selector_v2.k}")
    print("Algorithm limit: None")
    print(f"Solve rate: {sr_v2:.1%}  PAR10: {par10_v2:.2f}")
    print_sunny_schedules(predictions_v2, test_performance, budget, n=8)

    # 3. TSunny (algorithm limit tuning)
    print("\n" + "=" * 70)
    print("3. TSunny (algorithm limit tuning enabled)")
    print("=" * 70)
    selector_tsunny = SUNNY(k=5, use_tsunny=True, budget=budget)
    selector_tsunny.fit(train_features, train_performance)
    predictions_tsunny = cast(
        dict[str, list[tuple[str, float]]], selector_tsunny.predict(test_features)
    )
    sr_tsunny = compute_solve_rate(predictions_tsunny, test_performance, budget)
    par10_tsunny = running_time_selector_performance(
        predictions_tsunny,
        test_performance,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"k = {selector_tsunny.k}")
    print(f"Tuned algorithm limit = {selector_tsunny.tuned_algorithm_limit}")
    print(f"Solve rate: {sr_tsunny:.1%}  PAR10: {par10_tsunny:.2f}")
    print_sunny_schedules(predictions_tsunny, test_performance, budget, n=8)

    # 4. Hardcoded algorithm limit
    print("\n" + "=" * 70)
    print("4. SUNNY with hardcoded algorithm_limit=2")
    print("=" * 70)
    selector_hardcoded = SUNNY(k=5, algorithm_limit=2, budget=budget)
    selector_hardcoded.fit(train_features, train_performance)
    predictions_hardcoded = cast(
        dict[str, list[tuple[str, float]]], selector_hardcoded.predict(test_features)
    )
    sr_hardcoded = compute_solve_rate(predictions_hardcoded, test_performance, budget)
    par10_hardcoded = running_time_selector_performance(
        predictions_hardcoded,
        test_performance,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"k = {selector_hardcoded.k}")
    print(f"Hardcoded algorithm limit = {selector_hardcoded.algorithm_limit}")
    print(f"Solve rate: {sr_hardcoded:.1%}  PAR10: {par10_hardcoded:.2f}")
    print_sunny_schedules(predictions_hardcoded, test_performance, budget, n=8)

    # 5. Combined: k-tuning + algorithm limit tuning
    print("\n" + "=" * 70)
    print("5. Combined (k-tuning + algorithm limit tuning)")
    print("=" * 70)
    selector_combined = SUNNY(
        use_v2=True, use_tsunny=True, budget=budget, k_candidates=[3, 5, 7, 10, 15]
    )
    selector_combined.fit(train_features, train_performance)
    predictions_combined = cast(
        dict[str, list[tuple[str, float]]], selector_combined.predict(test_features)
    )
    sr_combined = compute_solve_rate(predictions_combined, test_performance, budget)
    par10_combined = running_time_selector_performance(
        predictions_combined,
        test_performance,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"Tuned k = {selector_combined.k}")
    print(f"Tuned algorithm limit = {selector_combined.tuned_algorithm_limit}")
    print(f"Solve rate: {sr_combined:.1%}  PAR10: {par10_combined:.2f}")
    print_sunny_schedules(predictions_combined, test_performance, budget, n=8)

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(
        f"{'Method':<30} {'Solve Rate':>12} {'PAR10':>12} {'k':>5} {'Algo Limit':>12}"
    )
    print("-" * 70)
    print(
        f"{'Basic SUNNY':<30} {sr:>11.1%} {par10:>12.2f} {selector.k:>5} {'None':>12}"
    )
    print(
        f"{'SUNNY-AS2':<30} {sr_v2:>11.1%} {par10_v2:>12.2f} {selector_v2.k:>5} {'None':>12}"
    )
    print(
        f"{'TSunny':<30} {sr_tsunny:>11.1%} {par10_tsunny:>12.2f} {selector_tsunny.k:>5} {selector_tsunny.tuned_algorithm_limit:>12}"
    )
    print(
        f"{'Hardcoded limit=2':<30} {sr_hardcoded:>11.1%} {par10_hardcoded:>12.2f} {selector_hardcoded.k:>5} {selector_hardcoded.algorithm_limit:>12}"
    )
    print(
        f"{'Combined tuning':<30} {sr_combined:>11.1%} {par10_combined:>12.2f} {selector_combined.k:>5} {selector_combined.tuned_algorithm_limit:>12}"
    )
