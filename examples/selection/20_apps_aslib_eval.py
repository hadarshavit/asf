import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asf.selectors.parallel_portfolio_selector import APPS
from asf.predictors.random_forest import RandomForestRegressorWrapper
from asf.scenario import aslib_reader
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def load_scenario(base_dir: str, name: str):
    scenario_path = os.path.join(base_dir, name)
    return aslib_reader.read_aslib_scenario(scenario_path)


def run_cv_evaluation(
    base_dir: str, scenario_name: str, p_values=(0.59, 0.82), folds=10
):
    data = load_scenario(base_dir, scenario_name)
    features, performance, budget = data[0], data[1], data[6]
    features = features.fillna(features.mean())
    perf = performance.fillna(budget * 10)

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Storage for detailed analysis
    results = {p: {"par10": [], "solve_rate": [], "sizes": []} for p in p_values}
    baseline_stats = {
        "sbs_par10": [],
        "vbs_par10": [],
        "sbs_solve_rate": [],
        "vbs_solve_rate": [],
        "sbs_name": [],
    }

    print(
        f"\n{'=' * 80}\nSCENARIO: {scenario_name} | Budget: {budget} | Algos: {len(perf.columns)}\n{'=' * 80}"
    )
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(features), 1):
        fold_start = time.time()
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        Y_train, Y_test = perf.iloc[train_idx], perf.iloc[test_idx]

        # Baselines (SBS/VBS)
        sbs_algo = Y_train.mean().idxmin()
        baseline_stats["sbs_name"].append(sbs_algo)

        sbs_par10 = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
        vbs_par10 = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)

        sbs_preds = {idx: [(sbs_algo, budget)] for idx in X_test.index}
        vbs_preds = {idx: [(Y_test.loc[idx].idxmin(), budget)] for idx in X_test.index}
        sbs_sr = compute_solve_rate(sbs_preds, Y_test, budget)
        vbs_sr = compute_solve_rate(vbs_preds, Y_test, budget)

        baseline_stats["sbs_par10"].append(sbs_par10)
        baseline_stats["vbs_par10"].append(vbs_par10)
        baseline_stats["sbs_solve_rate"].append(sbs_sr)
        baseline_stats["vbs_solve_rate"].append(vbs_sr)

        for p in p_values:
            sel = APPS(
                model_class=RandomForestRegressorWrapper,
                p_intersection=p,
                n_estimators_for_std=5,
            )
            sel.fit(X_train, Y_train)
            preds = sel.predict(X_test)

            # APPS already returns schedules in the correct format: {inst: [(algo, budget), ...]}
            sr_apps = compute_solve_rate(preds, Y_test, budget)
            par10_apps = running_time_selector_performance(
                preds,
                Y_test,
                budget=budget,
                par=10.0,
                return_per_instance=False,
            )

            results[p]["par10"].append(par10_apps)
            results[p]["solve_rate"].append(sr_apps)
            results[p]["sizes"].extend([len(port) for port in preds.values()])

        print(f"  > Fold {fold}/{folds} Completed ({time.time() - fold_start:.1f}s)")

    # --- FINAL AGGREGATION & REPORTING ---
    total_time = time.time() - start_time
    avg_sbs = np.mean(baseline_stats["sbs_par10"])
    avg_vbs = np.mean(baseline_stats["vbs_par10"])
    std_sbs = np.std(baseline_stats["sbs_par10"])
    avg_sbs_sr = np.mean(baseline_stats["sbs_solve_rate"])
    avg_vbs_sr = np.mean(baseline_stats["vbs_solve_rate"])

    print(f"\n{'#' * 30} FINAL REPORT {'#' * 30}")
    print(f"Total Execution Time: {total_time / 60:.2f} minutes")
    print(f"Common SBS Algorithms: {set(baseline_stats['sbs_name'])}")
    print(f"{'-' * 74}")
    print(
        f"{'Metric':<15} | {'Mean PAR10':<15} | {'Std Dev':<12} | {'Gap Closed':<12} | {'Solve-rate':<12}"
    )
    print(f"{'-' * 74}")
    print(
        f"{'SBS (Baseline)':<15} | {avg_sbs:<15.2f} | {std_sbs:<12.2f} | 0.00% | {avg_sbs_sr:<12.2%}"
    )
    print(
        f"{'VBS (Oracle)':<15} | {avg_vbs:<15.2f} | {'N/A':<12} | 100.00% | {avg_vbs_sr:<12.2%}"
    )
    print(f"{'-' * 74}")

    for p in p_values:
        p_par10 = np.mean(results[p]["par10"])
        p_std = np.std(results[p]["par10"])
        gap = (avg_sbs - p_par10) / (avg_sbs - avg_vbs) if avg_sbs != avg_vbs else 0
        p_sr = np.mean(results[p]["solve_rate"])

        sizes = results[p]["sizes"]
        print(
            f"APPS (p={p:<4.2f}) | {p_par10:<15.2f} | {p_std:<12.2f} | {gap:<12.2%} | {p_sr:<12.2%}"
        )
        print(
            f"   ↳ Portfolio Size: Avg={np.mean(sizes):.1f}, Max={np.max(sizes)}, Min={np.min(sizes)}"
        )
    print(f"{'#' * 74}\n")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aslib_data")
    # scenarios = {"IPC2018": (0.59, 0.82), "SAT11-INDU": (0.63, 0.82)}
    scenarios = {"SAT11-INDU": (0.63, 0.82)}

    for sc, p_vals in scenarios.items():
        run_cv_evaluation(base_dir, sc, p_vals, 3)
