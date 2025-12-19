import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from asf.selectors.rpc_selector import RPCSelector


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


def evaluate_solve_rate(preds: dict, perf: pd.DataFrame, budget: float) -> float:
    """Compute fraction of instances solved within budget by chosen algorithm."""
    solved = 0
    total = 0
    for inst, rec in preds.items():
        if not rec:
            continue
        algo, _ = rec[0]
        if algo is None:
            continue
        total += 1
        rt = perf.at[inst, algo]
        if not np.isnan(rt) and float(rt) <= budget:
            solved += 1
    return solved / total if total > 0 else 0.0


def evaluate_parallel_portfolio(
    preds: dict, true_perf: pd.DataFrame, budget: float
) -> float:
    """Solve-rate when running all algorithms in the portfolio in parallel."""
    solved = 0
    total = len(preds)

    for inst, algo_list in preds.items():
        if inst not in true_perf.index or not algo_list:
            continue
        instance_solved = False
        for algo in algo_list:
            rt = true_perf.loc[inst, algo]
            if not np.isnan(rt) and float(rt) <= budget:
                instance_solved = True
                break
        if instance_solved:
            solved += 1
    return solved / total if total > 0 else 0.0


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
    best_algo = ((Y_train <= budget).mean(axis=0)).idxmax()
    baseline_sr = float((Y_test[best_algo] <= budget).mean())
    oracle_sr = float((Y_test.min(axis=1) <= budget).mean())

    print(f"Best-single baseline ({best_algo}) solve-rate: {baseline_sr:.2%}")
    print(f"Oracle solve-rate: {oracle_sr:.2%}")
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
    sr_rf = evaluate_solve_rate(preds_rf, Y_test, budget)
    print(f"  Solve-rate: {sr_rf:.2%}")
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
    sr_dt = evaluate_solve_rate(preds_dt, Y_test, budget)
    print(f"  Solve-rate: {sr_dt:.2%}")
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

    # Top-3 parallel solve-rate
    acc_top3 = evaluate_parallel_portfolio(portfolios, Y_test, budget)
    print(f"  Parallel solve-rate (top-3): {acc_top3:.2%}")

    # Show first 10 portfolios and which algorithms solve
    print("\nDetailed portfolios (first 10 instances):")
    for inst in list(X_test.index)[:10]:
        portfolio = portfolios.get(inst, [])
        solvers = []
        for algo in portfolio:
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
