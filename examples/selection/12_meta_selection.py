import numpy as np
import pandas as pd
from typing import Any, cast

from asf.predictors.random_forest import (
    RandomForestClassifierWrapper as RandomForestClassifier,
)
from asf.selectors.meta_selector import MetaSelector
from asf.selectors.snnap import SNNAP
from asf.selectors.satzilla import SATzilla
from asf.selectors.isac import ISAC
from asf.selectors.multi_class import MultiClassClassifier
from asf.selectors.survival_analysis import SurvivalAnalysis

BASE_CLASSES = [SNNAP, SATzilla, ISAC]
META_CLASS = SATzilla


def make_challenging_data(n_instances=200, n_algorithms=6, seed=42, budget=200.0):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.uniform(0, 10, size=(n_instances, 4)),
        columns=[f"f{i}" for i in range(4)],  # type: ignore[arg-type]
        index=[f"inst_{i}" for i in range(n_instances)],  # type: ignore[arg-type]
    )

    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(10, 200) * (1 + 0.5 * (a % 2))
        noise = rng.normal(0, 30, size=n_instances)
        runtimes = np.maximum(5.0, bias + features["f0"] * rng.uniform(-5, 5) + noise)
        timeout_mask = rng.rand(n_instances) < (0.10 + 0.05 * (a % 3))
        runtimes[timeout_mask] = budget
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def evaluate_predictions(predictions: Any, true_perf: pd.DataFrame, budget: float):
    total = 0.0
    solved = 0
    n = len(true_perf)
    for inst in true_perf.index:
        sched = predictions.get(inst, [])
        if not sched:
            total += budget
            continue
        algo, _ = sched[0]
        rt = true_perf.loc[inst, algo]
        if pd.isna(rt) or rt >= budget:
            total += budget
        else:
            total += float(rt)
            solved += 1
    return total / n, solved / n


def run_base_selectors(base_selectors, train_X, train_Y, test_X, test_Y, budget):
    results = []
    for sel in base_selectors:
        name = sel.__class__.__name__
        try:
            sel.fit(train_X, train_Y)
            preds = sel.predict(test_X)
            avg_rt, solve_rate = evaluate_predictions(preds, test_Y, budget)
            results.append((name, avg_rt, solve_rate))
        except Exception as _:
            results.append((name, float("inf"), 0.0))
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
        base_selectors=meta_base_selectors, meta_selector=ISAC(), budget=budget
    )

    meta.fit(train_X, train_Y)
    meta_preds = meta.predict(test_X)
    meta_avg_rt, meta_solve_rate = evaluate_predictions(meta_preds, test_Y, budget)

    # print comparison
    print("=" * 60)
    print("MetaSelector vs Base Selectors")
    print("=" * 60)
    print(f"{'Selector':<20} {'Avg Runtime':>12} {'Solve Rate':>12}")
    print("-" * 46)
    for name, avg_rt, solve_rate in base_results:
        print(f"{name:<20} {avg_rt:12.2f}s {solve_rate:11.1%}")
    print("-" * 46)
    print(f"{'MetaSelector':<20} {meta_avg_rt:12.2f}s {meta_solve_rate:11.1%}")
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
