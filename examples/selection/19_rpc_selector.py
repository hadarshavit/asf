import numpy as np
import pandas as pd
from typing import cast
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from asf.selectors.rpc_selector import RPCSelector
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def make_data(n_instances=300, n_algorithms=6, seed=2, budget=200.0):
    """Generate synthetic features and per-algorithm runtime matrix.

    Runtimes are positive with some timeouts (> budget).
    """
    rng = np.random.RandomState(seed)

    # Instance features
    n_feats = 6
    features = pd.DataFrame(
        rng.normal(size=(n_instances, n_feats)),
        columns=[f"f{i}" for i in range(n_feats)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    # Algorithm runtimes derived from linear relation + noise + occasional timeouts
    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(20, 120) * (1 + 0.25 * (a % 2))
        coeff = rng.uniform(-3, 3, size=n_feats)
        noise = rng.normal(0, 10, size=n_instances)
        runtimes = np.clip(features.values @ coeff + bias + noise, 1.0, None)
        timeout_mask = rng.rand(n_instances) < (0.10 + 0.03 * (a % 3))
        runtimes[timeout_mask] = budget * 10
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def main():
    budget = 200.0
    X, Y = make_data(n_instances=300, n_algorithms=10, seed=2, budget=budget)

    # Train/test split
    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    print("=" * 70)
    print("RPC (Ranking by Pairwise Comparison) Example")
    print("=" * 70)
    print(f"Budget: {budget}s")
    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(f"Algorithms: {list(Y.columns)}")
    print()

    # Baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)

    print(f"Single Best Solver (SBS) PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_score:.2f}")
    print("-" * 70)

    # Test with RandomForest classifier (single best)
    print("RandomForestClassifier (n_estimators=50, top_n=1):")
    selector_rf = RPCSelector(
        classifier_class=RandomForestClassifier,
        n_estimators=50,
        random_state=42,
        budget=budget,
        top_n=1,
    )
    selector_rf.fit(X_train, Y_train)
    preds_rf = selector_rf.predict(X_test)
    assert isinstance(preds_rf, dict)
    preds_rf = cast(dict[str, list[tuple[str, float] | str]], preds_rf)

    sr_rf = compute_solve_rate(preds_rf, Y_test, budget)
    par10_rf = running_time_selector_performance(
        preds_rf, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_rf:.2%} | PAR10: {par10_rf:.2f}")
    print()

    # Test with DecisionTree classifier
    print("DecisionTreeClassifier (max_depth=10, top_n=1):")
    selector_dt = RPCSelector(
        classifier_class=DecisionTreeClassifier,
        classifier_kwargs={"max_depth": 10, "random_state": 42},
        budget=budget,
        top_n=1,
    )
    selector_dt.fit(X_train, Y_train)
    preds_dt = selector_dt.predict(X_test)
    assert isinstance(preds_dt, dict)
    preds_dt = cast(dict[str, list[tuple[str, float] | str]], preds_dt)

    sr_dt = compute_solve_rate(preds_dt, Y_test, budget)
    par10_dt = running_time_selector_performance(
        preds_dt, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_dt:.2%} | PAR10: {par10_dt:.2f}")
    print()

    # Parallel shortlist: top-3 algorithms
    print("-" * 70)
    print("RPC parallel shortlist (top_n=3) with RandomForest:")
    selector_rf_top3 = RPCSelector(
        classifier_class=RandomForestClassifier,
        n_estimators=50,
        random_state=42,
        budget=budget,
        top_n=3,
    )
    selector_rf_top3.fit(X_train, Y_train)
    portfolios = selector_rf_top3.predict(X_test)
    assert isinstance(portfolios, dict)
    portfolios = cast(dict[str, list[tuple[str, float] | str]], portfolios)
    # Convert shortlist to schedules with equal time slices summing to budget

    sr_top3 = compute_solve_rate(portfolios, Y_test, budget)
    par10_top3 = running_time_selector_performance(
        portfolios,
        Y_test,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"  Parallel shortlist solve-rate: {sr_top3:.2%} | PAR10: {par10_top3:.2f}")

    # Show first 10 portfolios and which algorithms solve
    print("\nDetailed portfolios (first 10 instances):")
    for inst in list(X_test.index)[:10]:
        portfolio = portfolios.get(str(inst), [])
        assert isinstance(portfolio, list)
        solvers = []
        if len(portfolio) > 0:
            for algo, _allocated in portfolio:
                rt = Y_test.at[inst, algo]
                if not np.isnan(rt) and float(rt) <= budget:
                    solvers.append(f"{algo}({rt:.1f}s)")
        solved_str = " ✓" if solvers else " ✗"
        print(f"{inst}: {portfolio}{solved_str}")
        if solvers:
            print(f"  └─ Solved by: {', '.join(solvers)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
