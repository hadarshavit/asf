import logging
import os
import pandas as pd
import numpy as np

from asf.metrics.baselines import running_time_closed_gap
from asf.scenario.aslib_reader import evaluate_selector
from asf.selectors.baselines import SingleBestSolver, VirtualBestSolver
from asf.selectors.hybrid_decision_tree import HARRIS
from asf.selectors.isac import ISAC
from asf.clustering.wrappers import KMeansWrapper
from asf.presolving import Static3S
from asf.selectors.selector_tuner import tune_selector
from functools import partial

# Configure logging to print debug logs for all loggers
logging.basicConfig(level=logging.INFO, force=True)


def run(selector, scenario, fold, base_path, use_HPO=False):
    if hasattr(selector, "__name__"):
        selector_name = selector.__name__
    elif hasattr(selector, "func"):  # for functools.partial
        selector_name = selector.func.__name__
    elif isinstance(selector, tuple):
        selector_name = selector[0].__name__
    else:
        selector_name = selector.__class__.__name__

    is_baseline = selector in (SingleBestSolver, VirtualBestSolver)
    if not is_baseline:
        print(f"Evaluating selector: {selector_name}")
        print(f"Evaluating scenario: {scenario}")
        print(f"Evaluating fold: {fold}")

    if is_baseline:
        score, result_selector, per_instance_scores = evaluate_selector(
            selector_class=selector,
            scenario_path=os.path.join(base_path, scenario),
            fold=fold,
            hpo_func=None,
            metric=running_time_closed_gap,
            return_per_instance=True,
        )
    else:
        score, result_selector, per_instance_scores = evaluate_selector(
            selector_class=selector,
            scenario_path=os.path.join(base_path, scenario),
            fold=fold,
            hpo_func=tune_selector if use_HPO else None,
            metric=running_time_closed_gap,
            hpo_kwargs={
                "runcount_limit": 500,
                "cv": 10,
                "smac_kwargs": lambda scenario: {
                    "overwrite": True,
                    "logging_level": False,
                },
                "output_dir": f"{os.path.dirname(__file__)}/results/smac_selectors/{scenario}_{selector_name}_fold{fold}_selector_tuning",
                "max_algorithm_pre_selector": 10,
                "pre_solving_class": Static3S
                if scenario != "OPENML-WEKA-2017"
                else None,
            },
            algorithm_pre_selector=None,
            return_per_instance=True,
        )

    if not is_baseline:
        print(f"Selector: {result_selector} Test score: {score}")

    # Save per-instance results with columns: scenario, fold, selector, instance, test_score
    results = [
        {
            "scenario": scenario,
            "fold": fold,
            "selector": selector_name,
            "instance": instance,
            "test_score": instance_score,  # Raw running time
        }
        for instance, instance_score in per_instance_scores.items()
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"results_{scenario}_per_instance.csv")
    if os.path.exists(path):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


def run_baseline(selector, scenario, fold, base_path):
    """Run baseline selectors (SBS, VBS) without HPO."""
    selector_name = selector.__name__

    print(f"Evaluating baseline selector: {selector_name}")
    print(f"Evaluating scenario: {scenario}")
    print(f"Evaluating fold: {fold}")

    # Run without HPO for baseline selectors
    score, result_selector, per_instance_scores = evaluate_selector(
        selector_class=selector,
        scenario_path=os.path.join(base_path, scenario),
        fold=fold,
        hpo_func=None,  # No HPO for baselines
        metric=running_time_closed_gap,
        return_per_instance=True,
    )

    print(f"Selector: {selector_name} Test score: {score}")

    # Save per-instance results
    results = [
        {
            "scenario": scenario,
            "fold": fold,
            "selector": selector_name,
            "instance": instance,
            "test_score": instance_score,
        }
        for instance, instance_score in per_instance_scores.items()
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"results_{scenario}_per_instance.csv")
    if os.path.exists(path):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


def report_results(scenario, results_dir=None):
    """Read and report results with gap analysis.

    Parameters
    ----------
    scenario : str
        Scenario name to report results for
    results_dir : str, optional
        Directory containing results CSV files. Defaults to bench/results/
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")

    csv_path = os.path.join(results_dir, f"results_{scenario}_per_instance.csv")
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # test_score now contains PAR10 scores per instance
    # Sum by selector and fold to get total PAR10
    par10_summary = df.groupby(["selector", "fold"])["test_score"].sum().reset_index()
    par10_summary = par10_summary.rename(columns={"test_score": "par10_total"})

    # Compute PAR10 baselines
    from asf.scenario.aslib_reader import read_aslib_scenario
    from asf.metrics.baselines import single_best_solver, virtual_best_solver

    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "aslib_data")
    scenario_path = os.path.join(base_path, scenario)

    # Load scenario
    (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    ) = read_aslib_scenario(scenario_path)

    # Compute PAR10 baselines per fold
    par10_per_fold = {}
    for fold in range(1, df["fold"].max() + 1):
        test_instance_ids = cv.index[cv["fold"] == fold].unique()
        y_test = performance.loc[test_instance_ids]

        sbs_par10 = single_best_solver(y_test, maximize=False, budget=budget, par=10.0)
        vbs_par10 = virtual_best_solver(y_test, maximize=False, budget=budget, par=10.0)

        par10_per_fold[fold] = {"SBS": sbs_par10, "VBS": vbs_par10}

    print(f"\n{'=' * 80}")
    print(f"RESULTS SUMMARY: {scenario}")
    print(f"{'=' * 80}")
    print(
        f"{'Selector':<25} | {'Mean PAR10':<15} | {'Gap Closed':<15} | {'Std Dev':<15}"
    )
    print(f"{'-' * 80}")

    # Baselines
    sbs_par10_values = [
        par10_per_fold[fold]["SBS"] for fold in sorted(par10_per_fold.keys())
    ]
    vbs_par10_values = [
        par10_per_fold[fold]["VBS"] for fold in sorted(par10_per_fold.keys())
    ]

    sbs_mean = np.mean(sbs_par10_values)
    vbs_mean = np.mean(vbs_par10_values)
    gap_denominator = sbs_mean - vbs_mean

    print(
        f"{'SingleBestSolver':<25} | {sbs_mean:<15.2f} | {'0.00%':<15} | {np.std(sbs_par10_values):<15.2f}"
    )
    print(
        f"{'VirtualBestSolver':<25} | {vbs_mean:<15.2f} | {'100.00%':<15} | {np.std(vbs_par10_values):<15.2f}"
    )
    print(f"{'-' * 80}")

    # Print other selectors
    for selector in sorted(df["selector"].unique()):
        if selector in ["SingleBestSolver", "VirtualBestSolver"]:
            continue

        selector_par10 = par10_summary[par10_summary["selector"] == selector][
            "par10_total"
        ].values
        if len(selector_par10) == 0:
            continue

        sel_mean_par10 = np.mean(selector_par10)
        sel_std_par10 = np.std(selector_par10)

        # Compute gap closed
        if abs(gap_denominator) > 1e-9:
            gap_closed = (sbs_mean - sel_mean_par10) / gap_denominator * 100
        else:
            gap_closed = 0.0

        print(
            f"{selector:<25} | {sel_mean_par10:<15.2f} | {gap_closed:<14.2f}% | {sel_std_par10:<15.2f}"
        )

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Configuration
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "aslib_data")
    scenarios = ["MIP-2016"]
    n_folds = 3

    selectors = [
        SingleBestSolver,
        VirtualBestSolver,
        partial(
            HARRIS,
            n_estimators=100,
            max_depth=4,
            min_samples_split=2,
            lambda_param=0.8,
            max_features="sqrt",
            max_thresholds=32,
            random_state=42,
        ),
        partial(
            ISAC,
            clusterer=KMeansWrapper,
            clusterer_kwargs={"n_clusters": 5},
            random_state=42,
        ),
    ]

    # Run evaluation locally
    print(f"\n{'=' * 80}")
    print("Starting selector benchmark")
    print(f"Base path: {base_path}")
    print(f"{'=' * 80}\n")

    for scenario in scenarios:
        # Clear old results for this scenario
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"results_{scenario}_per_instance.csv")
        if os.path.exists(results_file):
            os.remove(results_file)

        for selector in selectors:
            for fold in range(1, n_folds + 1):
                try:
                    run(selector, scenario, fold, base_path)
                except Exception as e:
                    print(f"Error evaluating {selector} on {scenario} fold {fold}: {e}")
                    import traceback

                    traceback.print_exc()

            if selector == SingleBestSolver or selector == VirtualBestSolver:
                print(
                    f"Completed baseline selector: {selector.__name__} for scenario: {scenario}\n"
                )
            elif hasattr(selector, "func") and hasattr(selector.func, "__name__"):
                print(
                    f"Completed selector: {selector.func.__name__} for scenario: {scenario}\n"
                )
            else:
                selector_name = getattr(selector, "__name__", str(selector))
                print(f"Completed selector: {selector_name} for scenario: {scenario}\n")

        # Report results after all selectors are evaluated for this scenario
        report_results(scenario)
