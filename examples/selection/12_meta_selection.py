import numpy as np
import pandas as pd
from typing import cast

from asf.predictors.random_forest import (
    RandomForestClassifierWrapper as RandomForestClassifier,
)
from asf.selectors.meta_selector import MetaSelector
from asf.selectors.snnap import SNNAP
from asf.selectors.satzilla import SATzilla
from asf.selectors.multi_class import MultiClassClassifier
from asf.selectors.survival_analysis import SurvivalAnalysis
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def make_challenging_data(n_instances=200, n_algorithms=6, seed=42, budget=200.0):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.uniform(0, 10, size=(n_instances, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=[f"inst_{i}" for i in range(n_instances)]
    )

    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(10, 200) * (1 + 0.5 * (a % 2))
        noise = rng.normal(0, 30, size=n_instances)
        runtimes = np.maximum(5.0, bias + features["f0"] * rng.uniform(-5, 5) + noise)
        timeout_mask = rng.rand(n_instances) < (0.10 + 0.05 * (a % 3))
        runtimes[timeout_mask] = budget * 10
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def run_base_selectors(base_selectors, train_X, train_Y, test_X, test_Y, budget):
    results = []
    for sel in base_selectors:
        name = sel.__class__.__name__
        try:
            sel.fit(train_X, train_Y)
            preds = sel.predict(test_X)
            if preds is None or (isinstance(preds, dict) and len(preds) == 0):
                results.append((name, 0.0, float("inf")))
                continue

            budgeted_preds = {
                inst: [(algo, budget) for algo, _ in sched]
                for inst, sched in preds.items()
            }

            solve_rate = compute_solve_rate(budgeted_preds, test_Y, budget)
            par10 = running_time_selector_performance(
                budgeted_preds,
                test_Y,
                budget=budget,
                par=10.0,
                return_per_instance=False,
            )
            results.append((name, solve_rate, par10))
        except Exception:
            results.append((name, 0.0, float("inf")))
    return results


def main():
    budget = 200.0
    features, performance = make_challenging_data(
        n_instances=200, n_algorithms=6, seed=1, budget=budget
    )

    n_train = int(0.7 * len(features))
    train_X = features.iloc[:n_train]
    train_Y = performance.iloc[:n_train]
    test_X = features.iloc[n_train:]
    test_Y = performance.iloc[n_train:]

    # Use ASF metrics for baselines
    sbs_score = single_best_solver(test_Y, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(test_Y, maximize=False, budget=budget, par=10.0)

    # instantiate fresh base selectors
    base_selectors = [
        SNNAP(),
        SATzilla(),
        SurvivalAnalysis(budget=budget),
        MultiClassClassifier(model_class=RandomForestClassifier),
    ]

    # evaluate each base selector individually
    base_results = run_base_selectors(
        base_selectors, train_X, train_Y, test_X, test_Y, budget
    )

    meta_base_selectors = [
        SNNAP(),
        SATzilla(),
        SurvivalAnalysis(budget=budget),
        MultiClassClassifier(model_class=RandomForestClassifier),
    ]
    meta = MetaSelector(
        base_selectors=meta_base_selectors, meta_selector=SATzilla(), budget=budget
    )

    meta.fit(train_X, train_Y)
    meta_preds = meta.predict(test_X)
    budgeted_meta_preds = {
        inst: [(algo, budget) for algo, _ in sched]
        for inst, sched in meta_preds.items()
    }
    meta_sr = compute_solve_rate(budgeted_meta_preds, test_Y, budget)
    meta_par10 = running_time_selector_performance(
        budgeted_meta_preds, test_Y, budget=budget, par=10.0, return_per_instance=False
    )

    # print comparison
    print("=" * 60)
    print("MetaSelector vs Base Selectors")
    print("=" * 60)
    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print()
    print(f"{'Selector':<20} {'Solve Rate':>12} {'PAR10':>12}")
    print("-" * 46)
    for name, solve_rate, par10 in base_results:
        print(f"{name:<20} {solve_rate:11.1%} {par10:12.2f}")
    print("-" * 46)
    print(f"{'MetaSelector':<20} {meta_sr:11.1%} {meta_par10:12.2f}")
    print("\nSample meta decisions (first 10 instances):")
    for inst in list(test_Y.index)[:10]:
        sched = cast(dict, meta_preds).get(inst, [])
        if not sched:
            print(f"{inst}: <no prediction>")
            continue
        algo, _ = sched[0]
        print(f"{inst}: chosen algorithm {algo}")


if __name__ == "__main__":
    main()
