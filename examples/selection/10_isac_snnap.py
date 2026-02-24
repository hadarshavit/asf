import numpy as np
import pandas as pd
from typing import cast
from sklearn.cluster import KMeans

from asf.selectors.isac import ISAC
from asf.selectors.snnap import SNNAP
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_data(n_instances=120, n_algorithms=5, seed=0):
    """
    Synthetic dataset.
    """
    rng = np.random.RandomState(seed)

    n_clusters = 4
    centers = rng.uniform(0, 10, size=(n_clusters, 2))

    # create roughly even assignment to clusters to avoid tiny clusters dominating
    base_assign = np.repeat(np.arange(n_clusters), n_instances // n_clusters)
    if base_assign.size < n_instances:
        extra = rng.choice(n_clusters, size=(n_instances - base_assign.size))
        cluster_ids = np.concatenate([base_assign, extra])
    else:
        cluster_ids = base_assign[:n_instances]
    rng.shuffle(cluster_ids)

    features = pd.DataFrame(
        centers[cluster_ids] + rng.normal(scale=0.8, size=(n_instances, 2)),
        columns=["f1", "f2"],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    # Create cluster-specific base performance for each algorithm
    base_by_cluster = rng.uniform(40, 80, size=(n_clusters, n_algorithms))
    global_algo_bias = rng.normal(0, 3.0, size=(n_algorithms,))

    for cid in range(n_clusters):
        favored = cid % n_algorithms
        base_by_cluster[cid, favored] -= rng.uniform(2.0, 6.0)

    algo_coeffs = rng.uniform(-1.0, 1.0, size=(n_algorithms, 2))

    perf = pd.DataFrame(index=features.index)
    for j in range(n_algorithms):
        vals = []
        for i in range(n_instances):
            cid = cluster_ids[i]
            base = base_by_cluster[cid, j]
            linear = features.iloc[i].values @ algo_coeffs[j]
            noise = rng.normal(0, 4.0)
            vals.append(base + global_algo_bias[j] + linear + noise)
        perf[f"algo{j + 1}"] = vals

    perf[perf < 5] = 5
    return features, perf


def print_sample(predictions, true_perf, n=8):
    print("\nSample predictions:")
    for inst in list(true_perf.index)[:n]:
        rec = predictions.get(inst, [(None, 0.0)])
        algo = rec[0][0]
        runtime = true_perf.loc[inst, algo] if algo in true_perf.columns else None
        print(f"{inst}: recommended = {algo} | runtime = {runtime}")


if __name__ == "__main__":
    # generate data
    features, performance = generate_data(n_instances=120, n_algorithms=5, seed=1)

    # split train/test
    n_train = int(0.7 * len(features))
    train_features = features.iloc[:n_train]
    train_perf = performance.iloc[:n_train]
    test_features = features.iloc[n_train:]
    test_perf = performance.iloc[n_train:]

    print("\nTest performance (head):")
    print(test_perf.head(10))

    budget = 60

    # Baselines
    sbs_score = single_best_solver(test_perf, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(test_perf, maximize=False, budget=budget, par=10.0)

    print(f"\nSingle Best Solver PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10: {vbs_score:.2f}")
    print()

    # Default ISAC (GMeans) - may fail on small synthetic data
    print("ISAC (GMeans):")
    try:
        selector = ISAC()
        selector.fit(train_features, train_perf)
        preds = selector.predict(test_features)
        assert isinstance(preds, dict)
        preds = cast(dict[str, list[tuple[str, float] | str]], preds)
        budgeted_preds: dict[str, list[tuple[str, float] | str]] = {}
        for inst, sched in preds.items():
            if isinstance(sched, list) and len(sched) > 0:
                budgeted_preds[str(inst)] = [(str(algo), budget) for algo, _ in sched]
            else:
                budgeted_preds[str(inst)] = []
        sr = compute_solve_rate(budgeted_preds, test_perf, budget)
        par10 = running_time_selector_performance(
            budgeted_preds,
            test_perf,
            budget=budget,
            par=10.0,
            return_per_instance=False,
        )
        print(f"  Solve-rate: {sr:.2%}, PAR10: {par10:.2f}")
        print_sample(preds, test_perf, n=8)
    except ValueError as e:
        print(f"  Skipped (clustering failed): {str(e)[:80]}")

    # ISAC with KMeans (example: 6 clusters)
    print("\nISAC (KMeans, n_clusters=6):")
    selector_km = ISAC(clusterer=KMeans, clusterer_kwargs={"n_clusters": 6})
    selector_km.fit(train_features, train_perf)
    preds_km = selector_km.predict(test_features)
    assert isinstance(preds_km, dict)
    preds_km = cast(dict[str, list[tuple[str, float] | str]], preds_km)
    budgeted_preds_km: dict[str, list[tuple[str, float] | str]] = {}
    for inst, sched in preds_km.items():
        if isinstance(sched, list) and len(sched) > 0:
            budgeted_preds_km[str(inst)] = [(str(algo), budget) for algo, _ in sched]
        else:
            budgeted_preds_km[str(inst)] = []
    sr_km = compute_solve_rate(budgeted_preds_km, test_perf, budget)
    par10_km = running_time_selector_performance(
        budgeted_preds_km, test_perf, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_km:.2%}, PAR10: {par10_km:.2f}")
    print_sample(preds_km, test_perf, n=8)

    # SNNAP (k-NN majority-vote)
    print("\nSNNAP (k=5):")
    selector_snnap = SNNAP(k=5)
    selector_snnap.fit(train_features, train_perf)
    preds_snnap = selector_snnap.predict(test_features)
    assert isinstance(preds_snnap, dict)
    preds_snnap = cast(dict[str, list[tuple[str, float] | str]], preds_snnap)
    budgeted_preds_snnap: dict[str, list[tuple[str, float] | str]] = {}
    for inst, sched in preds_snnap.items():
        if isinstance(sched, list) and len(sched) > 0:
            budgeted_preds_snnap[str(inst)] = [(str(algo), budget) for algo, _ in sched]
        else:
            budgeted_preds_snnap[str(inst)] = []
    sr_snnap = compute_solve_rate(budgeted_preds_snnap, test_perf, budget)
    par10_snnap = running_time_selector_performance(
        budgeted_preds_snnap,
        test_perf,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"  Solve-rate: {sr_snnap:.2%}, PAR10: {par10_snnap:.2f}")
    print_sample(preds_snnap, test_perf, n=8)
