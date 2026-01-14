import numpy as np
import pandas as pd

from asf.selectors.hybrid_decision_tree import HARRIS
from asf.metrics import (
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
    compute_solve_rate,
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
        runtimes[timeout_mask] = budget * 2
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def main():
    budget = 200.0
    X, Y = make_data(n_instances=300, n_algorithms=6, seed=2, budget=budget)

    # Train/test split
    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    print("=" * 70)
    print("HARRIS (Hybrid Ranking and Regression Forests) Example")
    print("=" * 70)
    print(f"Budget: {budget}s")
    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(f"Algorithms: {list(Y.columns)}")
    print()

    # Use ASF metrics for baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_solve_rate = float((Y_test.min(axis=1) <= budget).mean())

    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) solve-rate: {vbs_solve_rate:.2%}")
    print("-" * 70)

    # Try a couple of lambda balances (ranking vs regression)
    for lam in [0.2, 0.8]:
        selector = HARRIS(
            n_estimators=20,
            max_depth=5,
            min_samples_split=4,
            lambda_param=lam,
            max_features="sqrt",
            max_thresholds=10,
            random_state=42,
            budget=budget,
        )
        selector.fit(X_train, Y_train)
        preds = selector.predict(X_test)
        sr = compute_solve_rate(preds, Y_test, budget)
        par10 = running_time_selector_performance(
            preds, Y_test, budget=budget, par=10.0, return_per_instance=False
        )
        print(f"lambda={lam:.2f} -> solve-rate: {sr:.2%}, PAR10: {par10:.2f}")

    print("-" * 70)
    print("Detailed predictions (lambda=0.5, first 10 instances):")
    selector = HARRIS(
        n_estimators=20,
        max_depth=5,
        min_samples_split=4,
        lambda_param=0.5,
        max_features="sqrt",
        max_thresholds=10,
        random_state=42,
        budget=budget,
    )
    selector.fit(X_train, Y_train)
    preds = selector.predict(X_test)
    sr = compute_solve_rate(preds, Y_test, budget)
    par10 = running_time_selector_performance(
        preds, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"lambda=0.5 -> solve-rate: {sr:.2%}, PAR10: {par10:.2f}")
    print()

    for inst in list(X_test.index)[:10]:
        algo, _ = preds.get(inst, [(None, None)])[0]
        if algo is not None and algo in Y_test.columns:
            rt = Y_test.at[inst, algo]
            solved_str = " ✓" if rt <= budget else " ✗"
            print(f"{inst}: chosen={algo}, true_rt={rt:.1f}s{solved_str}")
        else:
            print(f"{inst}: chosen={algo} (invalid)")

    print("=" * 70)


if __name__ == "__main__":
    main()
