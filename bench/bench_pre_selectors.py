"""
Benchmark script for evaluating pre-selectors on ASlib scenarios with 10-fold CV.

This script benchmarks all available pre-selectors by measuring:
1. The quality of the selected algorithm subset (VBS score after pre-selection)
2. The time taken to perform pre-selection

Usage:
    python bench_pre_selectors.py [--scenarios SCENARIO1 SCENARIO2 ...] [--output results.csv]
"""

import argparse
import logging
import os
import time
import resource
import submitit
import pandas as pd

from asf.metrics.baselines import virtual_best_solver, single_best_solver
from asf.pre_selector import (
    BeamSearchPreSelector,
    GeneticAlgorithmPreSelector,
    MarginalContributionBasedPreSelector,
    OptimizePreSelection,
    RandomLocalSearchPreSelector,
    SBSPreSelector,
)
from asf.scenario.aslib_reader import read_aslib_scenario

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define all pre-selectors to benchmark
def get_pre_selectors(n_algorithms: int, metric, maximize: bool = False) -> dict:
    """
    Returns a dictionary of pre-selector configurations to benchmark.

    Args:
        n_algorithms: Number of algorithms to select.
        metric: Metric function to optimize.
        maximize: Whether to maximize or minimize the metric.

    Returns:
        Dict mapping pre-selector names to instantiated pre-selectors.
    """
    pre_selectors = {
        # Simple/Fast methods
        "SBS": SBSPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
        ),
        # Marginal contribution methods
        "MarginalContribution_Backward": MarginalContributionBasedPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            mode="backward",
        ),
        "MarginalContribution_Forward": MarginalContributionBasedPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            mode="forward",
        ),
        # Beam search
        "BeamSearch_w5": BeamSearchPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            beam_width=5,
        ),
        "BeamSearch_w10": BeamSearchPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            beam_width=10,
        ),
        # Random local search
        "RandomLocalSearch_r5": RandomLocalSearchPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            n_restarts=5,
            max_iterations=50,
            seed=42,
        ),
        "RandomLocalSearch_r10": RandomLocalSearchPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            n_restarts=10,
            max_iterations=100,
            seed=42,
        ),
        # Genetic algorithm
        "GeneticAlgorithm_small": GeneticAlgorithmPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            population_size=20,
            n_generations=50,
            seed=42,
        ),
        "GeneticAlgorithm_large": GeneticAlgorithmPreSelector(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
            population_size=50,
            n_generations=100,
            seed=42,
        ),
        # Optimization-based
        "OptimizeDE": OptimizePreSelection(
            metric=metric,
            n_algorithms=n_algorithms,
            maximize=maximize,
        ),
    }

    return pre_selectors


def benchmark_pre_selector(
    pre_selector,
    performance: pd.DataFrame,
    metric,
) -> tuple[float, float, float, list]:
    """
    Benchmark a single pre-selector on performance data.

    Args:
        pre_selector: Pre-selector instance.
        performance: Performance DataFrame.
        metric: Metric function.
        maximize: Whether to maximize.

    Returns:
        Tuple of (score, wall_time, cpu_time, selected_algorithms).
    """
    start_wall_time = time.time()
    start_cpu_time = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )
    try:
        selected_performance = pre_selector.fit_transform(performance)
        elapsed_wall_time = time.time() - start_wall_time
        end_cpu_time = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
        )
        elapsed_cpu_time = end_cpu_time - start_cpu_time

        # Calculate metric on selected subset
        score = metric(selected_performance)

        # Get selected algorithm names
        if isinstance(selected_performance, pd.DataFrame):
            selected_algorithms = list(selected_performance.columns)
        else:
            selected_algorithms = []

        return score, elapsed_wall_time, elapsed_cpu_time, selected_algorithms

    except Exception as e:
        elapsed_wall_time = time.time() - start_wall_time
        end_cpu_time = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
        )
        elapsed_cpu_time = end_cpu_time - start_cpu_time
        logger.error(f"Pre-selector failed: {e}")
        return float("nan"), elapsed_wall_time, elapsed_cpu_time, []


def run_fold_benchmark(
    scenario_path: str,
    n_algorithms: int = 5,
    par_factor: float = 10.0,
    output: str = "results",
) -> list[dict]:
    """
    Run benchmark for all pre-selectors on a single fold.

    Args:
        scenario_path: Path to ASlib scenario.
        n_algorithms: Number of algorithms to select.
        par_factor: PAR factor for penalization.

    Returns:
        List of result dictionaries.
    """
    for fold in range(1, 11):
        scenario_name = os.path.basename(scenario_path)

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
        ) = read_aslib_scenario(scenario_path, training_par_factor=par_factor)

        # Align indices
        common_idx = features.index.intersection(cv.index)
        performance = performance.loc[common_idx]
        cv = cv.loc[common_idx]

        # Get training performance for this fold
        train_instance_ids = cv.index[cv["fold"] != fold].unique()
        train_performance = performance.loc[train_instance_ids]

        # Define metric (VBS on training data)
        def vbs_metric(perf):
            return virtual_best_solver(
                perf, maximize=maximize, budget=budget, par=par_factor
            )

        # Calculate baselines on training data
        vbs_score = vbs_metric(train_performance)
        sbs_score = single_best_solver(
            train_performance, maximize=maximize, budget=budget, par=par_factor
        )

        n_total_algorithms = len(train_performance.columns)

        if n_algorithms > n_total_algorithms:
            return

        results = []

        # Get pre-selectors
        pre_selectors = get_pre_selectors(
            n_algorithms=n_algorithms,
            metric=vbs_metric,
            maximize=maximize,
        )

        # Add brute force only for small scenarios
        # if comb(n_total_algorithms, n_algorithms) <= 10_000_000:
        #     pre_selectors["BruteForce"] = BruteForcePreSelector(
        #         metric=vbs_metric,
        #         n_algorithms=n_algorithms,
        #         maximize=maximize,
        #     )

        for pre_selector_name, pre_selector in pre_selectors.items():
            logger.info(f"  Running {pre_selector_name}...")

            score, elapsed_wall_time, elapsed_cpu_time, selected_algorithms = (
                benchmark_pre_selector(
                    pre_selector,
                    train_performance,
                    vbs_metric,
                )
            )

            # Calculate relative performance (lower is better for minimization)
            # Normalized score: (score - VBS) / (SBS - VBS) for minimization
            if not maximize:
                if sbs_score != vbs_score:
                    normalized_score = (sbs_score - score) / (sbs_score - vbs_score)
                else:
                    normalized_score = 0.0
            else:
                if sbs_score != vbs_score:
                    normalized_score = (score - sbs_score) / (vbs_score - sbs_score)
                else:
                    normalized_score = 0.0

            results.append(
                {
                    "scenario": scenario_name,
                    "fold": fold,
                    "pre_selector": pre_selector_name,
                    "n_algorithms_selected": n_algorithms,
                    "n_algorithms_total": n_total_algorithms,
                    "score": score,
                    "vbs_score": vbs_score,
                    "sbs_score": sbs_score,
                    "normalized_score": normalized_score,
                    "wall_time_seconds": elapsed_wall_time,
                    "cpu_time_seconds": elapsed_cpu_time,
                    "selected_algorithms": ",".join(selected_algorithms)
                    if selected_algorithms
                    else "",
                    "maximize": maximize,
                }
            )

        if os.path.exists(path := output):
            pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
        else:
            pd.DataFrame(results).to_csv(path, header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark pre-selectors on ASlib scenarios"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/home/ni574034/asf/bench/aslib_data",
        help="Base path to ASlib scenarios",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "SAT12-INDU",
            "SAT20-MAIN",
            "TSP-LION2015",
            "QBF-2016",
            "MIP-2016",
            "MAXSAT19-UCMS",
            "CSP-Minizinc-Time-2016",
            "ASP-POTASSCO",
            "BNSL-2016",
            "GRAPHS-2015",
        ],
        help="List of scenario names to benchmark",
    )
    parser.add_argument(
        "--n-algorithms",
        type=int,
        nargs="+",
        default=[2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help="List of number of algorithms to select (e.g., --n-algorithms 2 5 10 15)",
    )
    parser.add_argument(
        "--par-factor",
        type=float,
        default=10.0,
        help="PAR factor for penalization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ni574034/asf/bench/results/pre_selector_benchmark.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    executor = submitit.AutoExecutor("logs", "slurm")
    executor.update_parameters(
        timeout_min=60 * 24 * 1,
        slurm_partition="c23ms",
        slurm_array_parallelism=1200,
        cpus_per_task=1,
        mem_gb=15.7 * 1,
        tasks_per_node=1,
        slurm_job_name="PRESELECTOR_BENCH",
        slurm_account="lect0117",
        slurm_setup=[
            "module load -q GCCcore/12.2.0",
            "module load -q Python/3.10.8",
            "source /home/ni574034/venvs/asf_env/bin/activate",
        ],
    )
    with executor.batch():
        for scenario in args.scenarios:
            scenario_path = os.path.join(args.base_path, scenario)

            for n_algorithms in args.n_algorithms:
                logger.info(f" n_algorithms={n_algorithms}")
                try:
                    executor.submit(
                        run_fold_benchmark,
                        scenario_path=scenario_path,
                        n_algorithms=n_algorithms,
                        par_factor=args.par_factor,
                        output=args.output,
                    )
                except Exception as e:
                    logger.error(f"Error on fold {e}")
