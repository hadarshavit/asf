import pandas as pd
import numpy as np
from asf.selectors.sunny import SUNNY


def generate_simple_data(n_instances=100, seed=0):
    """Generate more challenging data with timeouts and varying algorithm behavior."""
    np.random.seed(seed)

    # Create features with 3 dimensions
    features = pd.DataFrame(
        np.random.uniform(0, 10, size=(n_instances, 3)),
        columns=["size", "density", "complexity"],
        index=[f"inst_{i}" for i in range(n_instances)],
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


def evaluate_selector(
    selector, train_features, train_performance, test_features, test_performance, budget
):
    """Fit and evaluate a selector, returning average runtime and solve rate."""
    selector.fit(train_features, train_performance)
    predictions = selector.predict(test_features)

    total_runtime = 0.0
    solved_count = 0

    for instance in test_performance.index:
        schedule = predictions[instance]
        row = test_performance.loc[instance]
        time_used = 0
        solved = False

        for algo, t in schedule:
            runtime = row[algo]
            time_used += t
            if runtime <= budget and runtime <= time_used:
                total_runtime += runtime
                solved = True
                solved_count += 1
                break

        if not solved:
            total_runtime += budget

    avg_runtime = total_runtime / len(test_performance)
    solve_rate = solved_count / len(test_performance)

    return predictions, avg_runtime, solve_rate


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

    # 1. Basic SUNNY (no tuning)
    print("\n" + "=" * 70)
    print("1. Basic SUNNY (fixed k=5, no tuning)")
    print("=" * 70)
    selector = SUNNY(k=5, use_v2=False, budget=budget)
    predictions, avg_rt, solve_rate = evaluate_selector(
        selector,
        train_features,
        train_performance,
        test_features,
        test_performance,
        budget,
    )
    print(f"k = {selector.k}")
    print("Algorithm limit: None")
    print(f"Average runtime: {avg_rt:.2f}s")
    print(f"Solve rate: {solve_rate:.1%}")
    print_sunny_schedules(predictions, test_performance, budget, n=8)

    # 2. SUNNY-AS2 (k-tuning)
    print("\n" + "=" * 70)
    print("2. SUNNY-AS2 (k-tuning enabled)")
    print("=" * 70)
    selector_v2 = SUNNY(k=5, use_v2=True, budget=budget, k_candidates=[3, 5, 7, 10, 15])
    predictions_v2, avg_rt_v2, solve_rate_v2 = evaluate_selector(
        selector_v2,
        train_features,
        train_performance,
        test_features,
        test_performance,
        budget,
    )
    print(f"Tuned k = {selector_v2.k}")
    print("Algorithm limit: None")
    print(f"Average runtime: {avg_rt_v2:.2f}s")
    print(f"Solve rate: {solve_rate_v2:.1%}")
    print_sunny_schedules(predictions_v2, test_performance, budget, n=8)

    # 3. TSunny (algorithm limit tuning)
    print("\n" + "=" * 70)
    print("3. TSunny (algorithm limit tuning enabled)")
    print("=" * 70)
    selector_tsunny = SUNNY(k=5, use_tsunny=True, budget=budget)
    predictions_tsunny, avg_rt_tsunny, solve_rate_tsunny = evaluate_selector(
        selector_tsunny,
        train_features,
        train_performance,
        test_features,
        test_performance,
        budget,
    )
    print(f"k = {selector_tsunny.k}")
    print(f"Tuned algorithm limit = {selector_tsunny.tuned_algorithm_limit}")
    print(f"Average runtime: {avg_rt_tsunny:.2f}s")
    print(f"Solve rate: {solve_rate_tsunny:.1%}")
    print_sunny_schedules(predictions_tsunny, test_performance, budget, n=8)

    # 4. Hardcoded algorithm limit
    print("\n" + "=" * 70)
    print("4. SUNNY with hardcoded algorithm_limit=2")
    print("=" * 70)
    selector_hardcoded = SUNNY(k=5, algorithm_limit=2, budget=budget)
    predictions_hardcoded, avg_rt_hardcoded, solve_rate_hardcoded = evaluate_selector(
        selector_hardcoded,
        train_features,
        train_performance,
        test_features,
        test_performance,
        budget,
    )
    print(f"k = {selector_hardcoded.k}")
    print(f"Hardcoded algorithm limit = {selector_hardcoded.algorithm_limit}")
    print(f"Average runtime: {avg_rt_hardcoded:.2f}s")
    print(f"Solve rate: {solve_rate_hardcoded:.1%}")
    print_sunny_schedules(predictions_hardcoded, test_performance, budget, n=8)

    # 5. Combined: k-tuning + algorithm limit tuning
    print("\n" + "=" * 70)
    print("5. Combined (k-tuning + algorithm limit tuning)")
    print("=" * 70)
    selector_combined = SUNNY(
        use_v2=True, use_tsunny=True, budget=budget, k_candidates=[3, 5, 7, 10, 15]
    )
    predictions_combined, avg_rt_combined, solve_rate_combined = evaluate_selector(
        selector_combined,
        train_features,
        train_performance,
        test_features,
        test_performance,
        budget,
    )
    print(f"Tuned k = {selector_combined.k}")
    print(f"Tuned algorithm limit = {selector_combined.tuned_algorithm_limit}")
    print(f"Average runtime: {avg_rt_combined:.2f}s")
    print(f"Solve rate: {solve_rate_combined:.1%}")
    print_sunny_schedules(predictions_combined, test_performance, budget, n=8)

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(
        f"{'Method':<30} {'Avg Runtime':>12} {'Solve Rate':>12} {'k':>5} {'Algo Limit':>12}"
    )
    print("-" * 70)
    print(
        f"{'Basic SUNNY':<30} {avg_rt:>12.2f}s {solve_rate:>11.1%} {selector.k:>5} {'None':>12}"
    )
    print(
        f"{'SUNNY-AS2':<30} {avg_rt_v2:>12.2f}s {solve_rate_v2:>11.1%} {selector_v2.k:>5} {'None':>12}"
    )
    print(
        f"{'TSunny':<30} {avg_rt_tsunny:>12.2f}s {solve_rate_tsunny:>11.1%} {selector_tsunny.k:>5} {selector_tsunny.tuned_algorithm_limit:>12}"
    )
    print(
        f"{'Hardcoded limit=2':<30} {avg_rt_hardcoded:>12.2f}s {solve_rate_hardcoded:>11.1%} {selector_hardcoded.k:>5} {selector_hardcoded.algorithm_limit:>12}"
    )
    print(
        f"{'Combined tuning':<30} {avg_rt_combined:>12.2f}s {solve_rate_combined:>11.1%} {selector_combined.k:>5} {selector_combined.tuned_algorithm_limit:>12}"
    )
