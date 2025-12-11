import numpy as np
import pandas as pd

from asf.selectors.parallel_portfolio_selector import APPS
from asf.predictors.random_forest import RandomForestRegressorWrapper


def make_data(n_instances=200, n_algorithms=5, seed=1, budget=200.0):
    """Generate synthetic algorithm performance data."""
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.normal(size=(n_instances, 6)),
        columns=[f"f{i}" for i in range(6)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(20, 120) * (1 + 0.3 * (a % 2))
        coeff = rng.uniform(-3, 3, size=features.shape[1])
        noise = rng.normal(0, 10, size=n_instances)
        runtimes = np.clip((features.values @ coeff) + bias + noise, 1.0, None)
        timeout_mask = rng.rand(n_instances) < (0.12 + 0.03 * (a % 3))
        runtimes[timeout_mask] = budget * 2
        perf[f"algo{a + 1}"] = runtimes
    return features, perf


def evaluate_parallel_portfolio(preds, true_perf, budget):
    """
    Evaluate parallel portfolio performance.

    For each instance, check if ANY algorithm in the portfolio solves it.
    This simulates running all algorithms in parallel.
    """
    solved = 0
    total = len(preds)

    for inst, algo_list in preds.items():
        if inst not in true_perf.index or not algo_list:
            continue

        # Check if any algorithm in the portfolio solves the instance
        instance_solved = False
        for algo in algo_list:
            rt = true_perf.loc[inst, algo]
            if not np.isnan(rt) and float(rt) <= budget:
                instance_solved = True
                break

        if instance_solved:
            solved += 1

    return solved / total if total > 0 else 0.0


def compute_portfolio_stats(preds):
    """Compute statistics about portfolio sizes."""
    sizes = [len(portfolio) for portfolio in preds.values()]
    return {
        "mean": np.mean(sizes),
        "median": np.median(sizes),
        "min": np.min(sizes),
        "max": np.max(sizes),
    }


def main():
    budget = 200.0
    X, Y = make_data(n_instances=300, n_algorithms=6, seed=2, budget=budget)

    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    # Test different p_intersection values
    p_values = [0.01, 0.1, 0.3, 0.5]

    print("=" * 70)
    print("APPS (Automatic Parallel Portfolio Selector) Example")
    print("=" * 70)
    print(f"Budget: {budget}s")
    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(f"Algorithms: {list(Y.columns)}")
    print()

    # Baseline: best single algorithm
    solve_rates = ((Y_train <= budget).mean(axis=0)).to_dict()
    best_algo = max(solve_rates, key=solve_rates.get)
    baseline_acc = float((Y_test[best_algo] <= budget).mean())

    # Oracle: at least one algorithm solves it
    oracle = float((Y_test.min(axis=1) <= budget).mean())

    print(f"Best-single baseline ({best_algo}) solve-rate: {baseline_acc:.2%}")
    print(f"Oracle solve-rate: {oracle:.2%}")
    print()
    print("-" * 70)

    for p_int in p_values:
        sel = APPS(
            model_class=RandomForestRegressorWrapper,
            p_intersection=p_int,
            n_estimators_for_std=5,  # Lower for faster training in example
            random_state=42,
        )
        sel.fit(X_train, Y_train)
        preds = sel.predict(X_test)

        # Validate structure
        assert isinstance(preds, dict)
        for v in preds.values():
            assert isinstance(v, list)
            for algo in v:
                assert algo in list(Y.columns)

        acc = evaluate_parallel_portfolio(preds, Y_test, budget)
        stats = compute_portfolio_stats(preds)

        print(f"p_intersection = {p_int:.2f}")
        print(f"  Solve-rate: {acc:.2%}")
        print(
            f"  Portfolio size - mean: {stats['mean']:.2f}, "
            f"median: {stats['median']:.0f}, "
            f"range: [{stats['min']:.0f}, {stats['max']:.0f}]"
        )
        print()

    # Detailed example with p_intersection=0.1
    print("-" * 70)
    print("Detailed predictions (p_intersection=0.1, first 10 instances):")
    print()

    sel = APPS(
        model_class=RandomForestRegressorWrapper,
        p_intersection=0.1,
        n_estimators_for_std=5,
        random_state=42,
    )
    sel.fit(X_train, Y_train)
    preds = sel.predict(X_test)

    for inst in list(X_test.index)[:10]:
        portfolio = preds.get(inst, [])
        # Check which algorithms actually solve it
        solvers = []
        for algo in portfolio:
            rt = Y_test.loc[inst, algo]
            if not np.isnan(rt) and float(rt) <= budget:
                solvers.append(f"{algo}({rt:.1f}s)")

        solved_str = " ✓" if solvers else " ✗"
        print(f"{inst}: {portfolio}{solved_str}")
        if solvers:
            print(f"  └─ Solved by: {', '.join(solvers)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
