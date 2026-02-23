import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from asf.selectors.satzilla import SATzilla
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_data(n_instances=80, seed=0):
    np.random.seed(seed)
    features = pd.DataFrame(
        np.random.uniform(0, 10, size=(n_instances, 2)),
        columns=["size", "density"],
        index=[f"inst_{i}" for i in range(n_instances)],
    )
    # Each algorithm's performance is a different function of the features
    performance = pd.DataFrame(
        {
            "algo1": 400
            + 20 * features["size"]
            - 100 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo2": 20
            + 35 * features["size"]
            + 6 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo3": 90
            - 20 * features["size"]
            + 80 * features["density"]
            + np.random.normal(0, 5, n_instances),
            "algo4": 100
            - 120 * features["size"]
            + 250 * features["density"]
            + np.random.normal(0, 5, n_instances),
        },
        index=features.index,
    )
    performance[performance < 5] = 5
    return features, performance


def print_sample(preds, true_perf, n=10):
    print("\nSample predictions:")
    for inst in list(true_perf.index)[:n]:
        rec = preds.get(inst, [(None, float("inf"))])
        algo, _ = rec[0]
        rt = true_perf.loc[inst, algo] if (algo in true_perf.columns) else None
        print(f"{inst}: recommended={algo} | runtime={rt}")


if __name__ == "__main__":
    budget = 200
    features, perf = generate_data(n_instances=200, seed=2)

    train_X, test_X, train_perf, test_perf = train_test_split(
        features, perf, test_size=0.3, random_state=2
    )

    best_train = train_perf.min(axis=1)
    sat_labels = (best_train < (budget * 0.7)).tolist()

    selector = SATzilla()
    selector.budget = budget
    selector.fit(train_X, train_perf, sat_labels=sat_labels)

    print(test_perf.head(10))

    preds = selector.predict(test_X)
    assert isinstance(preds, dict)

    # Baselines
    sbs_score = single_best_solver(test_perf, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(test_perf, maximize=False, budget=budget, par=10.0)

    # Selector metrics
    sr = compute_solve_rate(preds, test_perf, budget)
    par10 = running_time_selector_performance(
        preds, test_perf, budget=budget, par=10.0, return_per_instance=False
    )

    print("\nSATzilla example")
    print("=" * 50)
    print(f"Budget: {budget}s | Train/Test: {len(train_X)}/{len(test_X)}")
    print(f"Single Best Solver (SBS) PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_score:.2f}")
    print(f"SATzilla solve-rate: {sr:.2%}")
    print(f"SATzilla PAR10: {par10:.2f}")

    print_sample(preds, test_perf, n=12)
