import numpy as np
import pandas as pd
from typing import cast

from asf.selectors.osl_linear import OSLLinearSelector
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def make_data(n_instances=200, n_algorithms=5, seed=1, budget=200.0):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.normal(size=(n_instances, 6)),
        columns=[f"f{i}" for i in range(6)],
        index=[f"inst_{i}" for i in range(n_instances)]
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

    sel = OSLLinearSelector(
        budget=budget, reg=1e-3, optimizer_method="L-BFGS-B", maxiter=500
    )
    sel.fit(X_train, Y_train)

    preds = sel.predict(X_test)

    # simple validation of structure
    assert isinstance(preds, dict)
    for v in cast(dict, preds).values():
        assert isinstance(v, list) and len(v) == 1
        algo, score = v[0]
        assert algo in list(Y.columns) or algo is None

    acc = compute_solve_rate(preds, Y_test, budget)

    # Use ASF metrics for baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    par10 = running_time_selector_performance(
        preds, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    oracle = float((Y_test.min(axis=1) <= budget).mean())

    print("=" * 60)
    print("OSLLinearSelector example")
    print("=" * 60)
    print(f"Budget: {budget}s")
    print(f"Test instances: {len(X_test)}")
    print()
    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print()
    print(f"OSL selector solve-rate: {acc:.2%}")
    print(f"OSL selector PAR10 Score: {par10:.2f}")
    print(f"Oracle solve-rate: {oracle:.2%}")
    print()
    print("Sample decisions (first 12):")
    for inst in list(X_test.index)[:12]:
        rec = preds.get(inst, [(None, None)])
        print(f"{inst}: chosen = {rec[0][0]} (pred_score={rec[0][1]})")  # type: ignore[index]
    print("=" * 60)


if __name__ == "__main__":
    main()
