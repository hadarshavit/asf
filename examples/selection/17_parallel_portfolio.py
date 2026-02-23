import numpy as np
import pandas as pd

from asf.selectors.parallel_portfolio_selector import APPS
from asf.predictors.random_forest import RandomForestRegressorWrapper
from asf.metrics import (
    single_best_solver,
    virtual_best_solver,
    compute_solve_rate,
    running_time_selector_performance,
)


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

    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)

    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print()
    print("-" * 70)

    for p_int in p_values:
        sel = APPS(
            model_class=RandomForestRegressorWrapper,
            p_intersection=p_int,
            n_estimators_for_std=5,
            random_state=42,
        )
        sel.fit(X_train, Y_train)
        preds = sel.predict(X_test)

        # Use ASF metrics for evaluation (handles parallel portfolios correctly)
        solve_rate = compute_solve_rate(preds, Y_test, budget)
        par10_score = running_time_selector_performance(
            preds, Y_test, budget=budget, par=10.0, return_per_instance=False
        )

        # Portfolio size statistics
        sizes = [len(preds[inst]) for inst in Y_test.index if inst in preds]

        print(f"p_intersection = {p_int:.2f}")
        print(f"  Solve-rate: {solve_rate:.2%}")
        print(f"  PAR10 Score: {par10_score:.2f}")
        print(
            f"  Portfolio size - mean: {np.mean(sizes):.2f}, "
            f"median: {np.median(sizes):.0f}, "
            f"range: [{np.min(sizes):.0f}, {np.max(sizes):.0f}]"
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
    assert isinstance(preds, dict)

    for inst in list(X_test.index)[:10]:
        portfolio_schedule = preds.get(inst, [])
        assert isinstance(portfolio_schedule, list)
        # Extract algorithm names from schedule
        portfolio_algos = [
            str(algo)
            for algo, _ in portfolio_schedule
            if isinstance(algo, (str, int, float))
        ]
        # Get min runtime across portfolio (parallel execution)
        portfolio_times = Y_test.loc[inst, portfolio_algos]
        min_time = portfolio_times.min()
        solvers = [
            f"{algo}({Y_test.loc[inst, algo]:.1f}s)"
            for algo in portfolio_algos
            if Y_test.loc[inst, algo] <= budget
        ]

        solved_str = " ✓" if min_time <= budget else " ✗"
        print(f"{inst}: {portfolio_algos}{solved_str} (min: {min_time:.1f}s)")
        if solvers:
            print(f"  └─ Solved by: {', '.join(solvers)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
