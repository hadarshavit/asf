import numpy as np
import pandas as pd

from asf.selectors.osl_linear import OSLLinearSelector


def make_data(n_instances=200, n_algorithms=5, seed=1, budget=200.0):
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


def evaluate(preds, true_perf, budget):
    solved = 0
    total = 0
    for inst, rec in preds.items():
        if inst not in true_perf.index:
            continue
        algo = rec[0][0]
        if algo is None:
            continue
        total += 1
        rt = true_perf.loc[inst, algo]
        if not np.isnan(rt) and float(rt) <= budget:
            solved += 1
    return solved / total if total > 0 else 0.0


def main():
    budget = 200.0
    X, Y = make_data(n_instances=300, n_algorithms=6, seed=2, budget=budget)

    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    sel = OSLLinearSelector(
        budget=budget, reg=1e-3, optimizer_method="L-BFGS-B", maxiter=500
    )
    sel.fit(X_train, Y_train)

    preds = sel.predict(X_test)

    # simple validation of structure
    assert isinstance(preds, dict)
    for v in preds.values():
        assert isinstance(v, list) and len(v) == 1
        algo, score = v[0]
        assert algo in list(Y.columns) or algo is None

    acc = evaluate(preds, Y_test, budget)
    # baseline: best single algorithm on training (by solve-rate on train)
    solve_rates = ((Y_train <= budget).mean(axis=0)).to_dict()
    best_algo = max(solve_rates, key=solve_rates.get)
    baseline_preds = {idx: [(best_algo, 0.0)] for idx in X_test.index}
    baseline_acc = evaluate(baseline_preds, Y_test, budget)

    # oracle (upper bound)
    oracle = float((Y_test.min(axis=1) <= budget).mean())

    print("=" * 60)
    print("OSLLinearSelector example")
    print("=" * 60)
    print(f"Budget: {budget}s")
    print(f"Test instances: {len(X_test)}")
    print()
    print(f"OSL selector solve-rate (<=budget): {acc:.2%}")
    print(f"Best-single baseline ({best_algo}) solve-rate: {baseline_acc:.2%}")
    print(f"Oracle solve-rate: {oracle:.2%}")
    print()
    print("Sample decisions (first 12):")
    for inst in list(X_test.index)[:12]:
        rec = preds.get(inst, [(None, None)])
        print(f"{inst}: chosen = {rec[0][0]} (pred_score={rec[0][1]})")
    print("=" * 60)


if __name__ == "__main__":
    main()
