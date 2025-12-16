import numpy as np
import pandas as pd

from asf.selectors.hybrid_decision_tree import HARRIS


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

    # Baselines
    best_algo = ((Y_train <= budget).mean(axis=0)).idxmax()
    baseline_sr = float((Y_test[best_algo] <= budget).mean())
    oracle_sr = float((Y_test.min(axis=1) <= budget).mean())

    print(f"Best-single baseline ({best_algo}) solve-rate: {baseline_sr:.2%}")
    print(f"Oracle solve-rate: {oracle_sr:.2%}")
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
        sr = evaluate_solve_rate(preds, Y_test, budget)
        print(f"lambda={lam:.2f} -> solve-rate: {sr:.2%}")

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
    sr = evaluate_solve_rate(preds, Y_test, budget)
    print(f"lambda={lam:.2f} -> solve-rate: {sr:.2%}")

    for inst in list(X_test.index)[:10]:
        algo, _ = preds.get(inst, [(None, None)])[0]
        rt = Y_test.at[inst, algo] if algo is not None else float("nan")
        print(f"{inst}: chosen={algo}, true_rt={rt:.1f}s")

    print("=" * 70)


if __name__ == "__main__":
    main()
