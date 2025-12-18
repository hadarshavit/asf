import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asf.selectors.parallel_portfolio_selector import APPS
from asf.predictors.random_forest import RandomForestRegressorWrapper
from asf.scenario import aslib_reader


def load_scenario(base_dir: str, name: str):
    scenario_path = os.path.join(base_dir, name)
    return aslib_reader.read_aslib_scenario(scenario_path)


def compute_metrics(preds: dict, perf: pd.DataFrame, budget: float):
    runtimes, par10, mcp = [], [], []
    best_rt_per_inst = perf.min(axis=1)

    for inst, algos in preds.items():
        if inst not in perf.index or not algos:
            continue

        inst_perf_row = perf.loc[inst]
        selected_rts = [inst_perf_row[a] for a in algos if a in inst_perf_row.index]
        rt_sel = float(np.min(selected_rts)) if selected_rts else float("inf")

        runtimes.append(min(rt_sel, budget))
        par10.append(rt_sel if rt_sel <= budget else 10 * budget)
        mcp.append(max(0, rt_sel - best_rt_per_inst.get(inst, budget)))

    return {"runtime": np.mean(runtimes), "par10": np.mean(par10), "mcp": np.mean(mcp)}


def run_cv_evaluation(base_dir: str, scenario_name: str, p_values=(0.59, 0.82)):
    data = load_scenario(base_dir, scenario_name)
    features, performance, budget = data[0], data[1], data[6]
    features = features.fillna(features.mean())
    perf = performance.fillna(budget * 10)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Storage for detailed analysis
    results = {p: {"par10": [], "mcp": [], "sizes": []} for p in p_values}
    baseline_stats = {"sbs_par10": [], "vbs_par10": [], "sbs_name": []}

    print(
        f"\n{'=' * 80}\nSCENARIO: {scenario_name} | Budget: {budget} | Algos: {len(perf.columns)}\n{'=' * 80}"
    )
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(features), 1):
        fold_start = time.time()
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        Y_train, Y_test = perf.iloc[train_idx], perf.iloc[test_idx]

        # SBS Calculation
        sbs_algo = Y_train.mean().idxmin()
        baseline_stats["sbs_name"].append(sbs_algo)

        m_sbs = compute_metrics(
            {idx: [sbs_algo] for idx in X_test.index}, Y_test, budget
        )
        m_vbs = compute_metrics(
            {idx: [Y_test.loc[idx].idxmin()] for idx in X_test.index}, Y_test, budget
        )

        baseline_stats["sbs_par10"].append(m_sbs["par10"])
        baseline_stats["vbs_par10"].append(m_vbs["par10"])

        for p in p_values:
            sel = APPS(
                model_class=RandomForestRegressorWrapper,
                p_intersection=p,
                n_estimators_for_std=5,
            )
            sel.fit(X_train, Y_train)
            preds = sel.predict(X_test)

            m_apps = compute_metrics(preds, Y_test, budget)
            results[p]["par10"].append(m_apps["par10"])
            results[p]["mcp"].append(m_apps["mcp"])
            results[p]["sizes"].extend([len(port) for port in preds.values()])

        print(f"  > Fold {fold}/10 Completed ({time.time() - fold_start:.1f}s)")

    # --- FINAL AGGREGATION & REPORTING ---
    total_time = time.time() - start_time
    avg_sbs = np.mean(baseline_stats["sbs_par10"])
    avg_vbs = np.mean(baseline_stats["vbs_par10"])
    std_sbs = np.std(baseline_stats["sbs_par10"])

    print(f"\n{'#' * 30} FINAL REPORT {'#' * 30}")
    print(f"Total Execution Time: {total_time / 60:.2f} minutes")
    print(f"Common SBS Algorithms: {set(baseline_stats['sbs_name'])}")
    print(f"{'-' * 74}")
    print(f"{'Metric':<15} | {'Mean PAR10':<15} | {'Std Dev':<12} | {'Gap Closed':<12}")
    print(f"{'-' * 74}")
    print(f"{'SBS (Baseline)':<15} | {avg_sbs:<15.2f} | {std_sbs:<12.2f} | 0.00%")
    print(f"{'VBS (Oracle)':<15} | {avg_vbs:<15.2f} | {'N/A':<12} | 100.00%")
    print(f"{'-' * 74}")

    for p in p_values:
        p_par10 = np.mean(results[p]["par10"])
        p_std = np.std(results[p]["par10"])
        p_mcp = np.mean(results[p]["mcp"])
        gap = (avg_sbs - p_par10) / (avg_sbs - avg_vbs) if avg_sbs != avg_vbs else 0

        sizes = results[p]["sizes"]
        print(f"APPS (p={p:<4.2f}) | {p_par10:<15.2f} | {p_std:<12.2f} | {gap:<12.2%}")
        print(
            f"   ↳ MCP: {p_mcp:.2f} | Portfolio Size: Avg={np.mean(sizes):.1f}, Max={np.max(sizes)}, Min={np.min(sizes)}"
        )
    print(f"{'#' * 74}\n")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aslib_data")
    # scenarios = {"IPC2018": (0.59, 0.82), "SAT11-INDU": (0.63, 0.82)}
    scenarios = {"SAT11-INDU": (0.63, 0.82)}

    for sc, p_vals in scenarios.items():
        run_cv_evaluation(base_dir, sc, p_vals)
