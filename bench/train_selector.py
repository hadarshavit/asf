import logging
import os
import submitit
import pandas as pd

from asf.metrics.baselines import virtual_best_solver
from asf.pre_selector import MarginalContributionBasedPreSelector
from asf.predictors import RandomForestClassifierWrapper, RandomForestRegressorWrapper
from asf.scenario.aslib_reader import evaluate_selector
from asf.selectors import (
    MultiClassClassifier,
    PairwiseClassifier,
    PerformanceModel,
    PairwiseRegressor,
    SingleBestSolver,
    VirtualBestSolver,
)
from asf.presolving import Static3S
from asf.selectors.selector_tuner import tune_selector

# Configure logging to print debug logs for all loggers
logging.basicConfig(level=logging.INFO, force=True)


def run(selector, scenario, fold, base_path="/home/shavit/asf/paper/aslib_data"):
    if hasattr(selector, "__name__"):
        selector_name = selector.__name__
    elif hasattr(selector, "func"):  # for functools.partial
        selector_name = selector.func.__name__
    elif isinstance(selector, tuple):
        selector_name = selector[0].__name__
    else:
        selector_name = selector.__class__.__name__

    print(f"Evaluating selector: {selector_name}")
    print(f"Evaluating scenario: {scenario}")
    print(f"Evaluating fold: {fold}")

    if selector == SingleBestSolver or selector == VirtualBestSolver:
        score, result_selector, per_instance_scores = evaluate_selector(
            selector_class=selector,
            scenario_path=os.path.join(base_path, scenario),
            fold=fold,
            hpo_func=None,
            return_per_instance=True,
        )
    else:
        score, result_selector, per_instance_scores = evaluate_selector(
            selector_class=selector,
            scenario_path=os.path.join(base_path, scenario),
            fold=fold,
            hpo_func=tune_selector,
            hpo_kwargs={
                "runcount_limit": 500,
                "cv": 10,
                "smac_kwargs": lambda scenario: {
                    "overwrite": True,
                    "logging_level": False,
                },
                "output_dir": f"/home/ni574034/asf/bench/results/smac_selectors/{scenario}_{selector_name}_fold{fold}_selector_tuning",
                "max_algorithm_pre_selector": 10,
                "pre_solving_class": Static3S
                if scenario != "OPENML-WEKA-2017"
                else None,
            },
            algorithm_pre_selector=(
                MarginalContributionBasedPreSelector,
                {"metric": virtual_best_solver, "mode": "forward"},
            ),
            return_per_instance=True,
        )

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

    path = f"/home/ni574034/asf/bench/results/results_{scenario}_per_instance.csv"
    if os.path.exists(path):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


def run_baseline(
    selector, scenario, fold, base_path="/home/shavit/asf/paper/aslib_data"
):
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

    path = f"/home/ni574034/asf/bench/results/results_{scenario}_per_instance.csv"
    if os.path.exists(path):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


if __name__ == "__main__":
    selectors = [
        (MultiClassClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (PairwiseClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (PairwiseRegressor, {"model_class": [RandomForestRegressorWrapper]}),
        (PerformanceModel, {"model_class": [RandomForestRegressorWrapper]}),
        SingleBestSolver,
        VirtualBestSolver,
    ]

    executor = submitit.AutoExecutor("logs", "slurm")
    executor.update_parameters(
        timeout_min=60 * 24 * 1,
        slurm_partition="c23ms",
        slurm_array_parallelism=1200,
        cpus_per_task=1,
        mem_gb=15.7 * 1,
        tasks_per_node=1,
        slurm_job_name="PRESOLVER_BENCH",
        slurm_account="lect0117",
        slurm_setup=[
            "module load GCCcore/12.2.0",
            "module load Python/3.10.8",
            "source /home/ni574034/venvs/asf_env/bin/activate",
        ],
    )

    scenarios = [
        "ASP-POTASSCO",
        "BNSL-2016",
        "CPMP-2015",
        "CSP-2010",
        "CSP-MZN-2013",
        "CSP-Minizinc-Time-2016",
        "GLUHACK-2018",
        "MAXSAT12-PMS",
        "MAXSAT15-PMS-INDU",
        "QBF-2011",
        "SAT03-16_INDU",
        "SAT12-INDU",
        "SAT18-EXP",
    ]

    with executor.batch():
        for selector in selectors:
            for scenario in scenarios:
                for fold in range(1, 11):
                    executor.submit(
                        run,
                        selector,
                        scenario,
                        fold,
                        base_path="/home/ni574034/asf/paper/aslib_data",
                    )
