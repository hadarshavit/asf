"""
Benchmark script for evaluating presolvers on ASlib scenarios with 10-fold CV.

This script benchmarks all available presolvers by measuring:
1. The quality of the computed schedule (instances solved, total runtime)
2. The time taken to compute the schedule

Usage:
    python bench_presolvers.py [--scenarios SCENARIO1 SCENARIO2 ...] [--output results.csv]
"""

import argparse
import logging
import os
import time
import resource
import submitit
import pandas as pd

from asf.metrics.baselines import virtual_best_solver, single_best_solver
from asf.presolving import (
    ASAPv2,
    GreedyPresolver,
    Static3S,
    SubmodularPresolver,
)
from asf.scenario.aslib_reader import read_aslib_scenario

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Try to import optional presolvers
try:
    from asf.presolving import Aspeed

    ASPEED_AVAILABLE = True
except ImportError:
    ASPEED_AVAILABLE = False
    logger.warning("Aspeed presolver not available (clingo not installed)")


def get_presolvers(budget: float, maximize: bool = False) -> dict:
    """
    Returns a dictionary of presolver configurations to benchmark.

    Args:
        budget: Time budget for the presolver schedule.
        maximize: Whether to maximize or minimize the metric.

    Returns:
        Dict mapping presolver names to instantiated presolvers.
    """
    presolvers = {
        # Greedy presolver variants
        "Greedy_cutoff5": GreedyPresolver(
            budget=budget,
            cutoff_per_solver=5.0,
            max_presolvers=3,
            min_coverage=0.01,
            maximize=maximize,
        ),
        "Greedy_cutoff10": GreedyPresolver(
            budget=budget,
            cutoff_per_solver=10.0,
            max_presolvers=3,
            min_coverage=0.01,
            maximize=maximize,
        ),
        "Greedy_cutoff5_max5": GreedyPresolver(
            budget=budget,
            cutoff_per_solver=5.0,
            max_presolvers=5,
            min_coverage=0.01,
            maximize=maximize,
        ),
        # Submodular presolver variants
        "Submodular_default": SubmodularPresolver(
            budget=budget,
            time_discretization=[1.0, 2.0, 5.0, 10.0, 20.0],
            max_actions=10,
            maximize=maximize,
        ),
        "Submodular_fine": SubmodularPresolver(
            budget=budget,
            time_discretization=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
            max_actions=10,
            maximize=maximize,
        ),
        "Submodular_coarse": SubmodularPresolver(
            budget=budget,
            time_discretization=[5.0, 10.0, 20.0],
            max_actions=5,
            maximize=maximize,
        ),
        # ASAP v2 variants
        "ASAPv2_small": ASAPv2(
            runcount_limit=100.0,
            budget=budget,
            maximize=maximize,
            de_popsize=10,
            seed=42,
        ),
        "ASAPv2_medium": ASAPv2(
            runcount_limit=500.0,
            budget=budget,
            maximize=maximize,
            de_popsize=15,
            seed=42,
        ),
        "ASAPv2_large": ASAPv2(
            runcount_limit=1000.0,
            budget=budget,
            maximize=maximize,
            de_popsize=20,
            seed=42,
        ),
        # Static3S (requires pulp)
        "Static3S_default": Static3S(
            budget=budget,
            max_candidates_per_solver=20,
        ),
        "Static3S_fine": Static3S(
            budget=budget,
            max_candidates_per_solver=50,
        ),
    }

    # Add Aspeed if available
    if ASPEED_AVAILABLE:
        presolvers["Aspeed_default"] = Aspeed(
            budget=budget,
            aspeed_cutoff=60,
            maximize=maximize,
            cores=1,
        )

    return presolvers


def evaluate_schedule(
    schedule: list[tuple[str, float]],
    performance: pd.DataFrame,
    budget: float,
    par_factor: float,
    maximize: bool,
) -> dict:
    """
    Evaluate a presolver schedule on performance data.

    Args:
        schedule: List of (algorithm, time) tuples.
        performance: Performance DataFrame.
        budget: Total scenario budget.
        par_factor: PAR factor for penalization.
        maximize: Whether to maximize.

    Returns:
        Dictionary with evaluation metrics.
    """
    n_instances = len(performance)
    n_solved = 0
    total_runtime = 0.0
    schedule_cost = sum(t for _, t in schedule)

    for instance in performance.index:
        instance_runtime = 0.0
        solved = False

        for algo, allocated_time in schedule:
            if algo not in performance.columns:
                continue

            algo_runtime = performance.loc[instance, algo]

            if maximize:
                # For maximization, "solved" means performance >= some threshold
                # This is less common for presolving
                if algo_runtime >= allocated_time:
                    solved = True
                    instance_runtime += allocated_time
                    break
                else:
                    instance_runtime += allocated_time
            else:
                # For minimization (runtime), "solved" means runtime <= allocated_time
                if algo_runtime <= allocated_time:
                    solved = True
                    instance_runtime += algo_runtime
                    break
                else:
                    instance_runtime += allocated_time

        if solved:
            n_solved += 1
            total_runtime += instance_runtime
        else:
            # Instance not solved by presolver - penalize
            total_runtime += schedule_cost + budget * par_factor

    return {
        "n_solved": n_solved,
        "solve_rate": n_solved / n_instances if n_instances > 0 else 0.0,
        "total_runtime": total_runtime,
        "avg_runtime": total_runtime / n_instances if n_instances > 0 else 0.0,
        "schedule_cost": schedule_cost,
    }


def benchmark_presolver(
    presolver,
    features: pd.DataFrame,
    performance: pd.DataFrame,
    budget: float,
    par_factor: float,
    maximize: bool,
) -> tuple[dict, float, float, list]:
    """
    Benchmark a single presolver on performance data.

    Args:
        presolver: Presolver instance.
        features: Features DataFrame.
        performance: Performance DataFrame.
        budget: Scenario budget.
        par_factor: PAR factor.
        maximize: Whether to maximize.

    Returns:
        Tuple of (eval_metrics, wall_time, cpu_time, schedule).
    """
    start_wall_time = time.time()
    start_cpu_time = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )
    try:
        presolver.fit(features, performance)
        schedule = presolver.predict()

        elapsed_wall_time = time.time() - start_wall_time
        end_cpu_time = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
        )
        elapsed_cpu_time = end_cpu_time - start_cpu_time

        # Evaluate the schedule
        eval_metrics = evaluate_schedule(
            schedule, performance, budget, par_factor, maximize
        )

        return eval_metrics, elapsed_wall_time, elapsed_cpu_time, schedule

    except Exception as e:
        elapsed_wall_time = time.time() - start_wall_time
        end_cpu_time = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime
        )
        elapsed_cpu_time = end_cpu_time - start_cpu_time
        logger.error(f"Presolver failed: {e}")
        return (
            {
                "n_solved": 0,
                "solve_rate": 0.0,
                "total_runtime": float("nan"),
                "avg_runtime": float("nan"),
                "schedule_cost": 0.0,
            },
            elapsed_wall_time,
            elapsed_cpu_time,
            [],
        )


def run_fold_benchmark(
    scenario_path: str,
    fold: int,
    presolver_budget: float = 30.0,
    par_factor: float = 10.0,
    output: str = "results",
) -> list[dict]:
    """
    Run benchmark for all presolvers on a single fold.

    Args:
        scenario_path: Path to ASlib scenario.
        fold: Fold number (1-10).
        presolver_budget: Time budget for presolver schedules.
        par_factor: PAR factor for penalization.

    Returns:
        List of result dictionaries.
    """
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
    features = features.loc[common_idx]
    performance = performance.loc[common_idx]
    cv = cv.loc[common_idx]

    # Get training and test data for this fold
    train_instance_ids = cv.index[cv["fold"] != fold].unique()
    test_instance_ids = cv.index[cv["fold"] == fold].unique()

    train_features = features.loc[train_instance_ids]
    train_performance = performance.loc[train_instance_ids]
    test_performance = performance.loc[test_instance_ids]

    # Calculate baselines on test data
    vbs_score = virtual_best_solver(
        test_performance, maximize=maximize, budget=budget, par=par_factor
    )
    sbs_score = single_best_solver(
        test_performance, maximize=maximize, budget=budget, par=par_factor
    )

    n_total_algorithms = len(train_performance.columns)
    n_train_instances = len(train_performance)
    n_test_instances = len(test_performance)

    results = []

    # Get presolvers
    presolvers = get_presolvers(
        budget=presolver_budget,
        maximize=maximize,
    )

    for presolver_name, presolver in presolvers.items():
        logger.info(f"  Running {presolver_name}...")

        # Fit on training data
        eval_metrics, elapsed_wall_time, elapsed_cpu_time, schedule = (
            benchmark_presolver(
                presolver,
                train_features,
                train_performance,
                budget,
                par_factor,
                maximize,
            )
        )

        # Evaluate on test data
        if schedule:
            test_eval = evaluate_schedule(
                schedule, test_performance, budget, par_factor, maximize
            )
        else:
            test_eval = {
                "n_solved": 0,
                "solve_rate": 0.0,
                "total_runtime": float("nan"),
                "avg_runtime": float("nan"),
                "schedule_cost": 0.0,
            }

        # Format schedule for output
        schedule_str = (
            ";".join([f"{algo}:{t:.2f}" for algo, t in schedule]) if schedule else ""
        )

        results.append(
            {
                "scenario": scenario_name,
                "fold": fold,
                "presolver": presolver_name,
                "presolver_budget": presolver_budget,
                "n_algorithms_total": n_total_algorithms,
                "n_train_instances": n_train_instances,
                "n_test_instances": n_test_instances,
                # Training evaluation
                "train_n_solved": eval_metrics["n_solved"],
                "train_solve_rate": eval_metrics["solve_rate"],
                "train_total_runtime": eval_metrics["total_runtime"],
                "train_avg_runtime": eval_metrics["avg_runtime"],
                # Test evaluation
                "test_n_solved": test_eval["n_solved"],
                "test_solve_rate": test_eval["solve_rate"],
                "test_total_runtime": test_eval["total_runtime"],
                "test_avg_runtime": test_eval["avg_runtime"],
                # Baselines (on test data)
                "vbs_score": vbs_score,
                "sbs_score": sbs_score,
                # Timing
                "wall_time_seconds": elapsed_wall_time,
                "cpu_time_seconds": elapsed_cpu_time,
                # Schedule info
                "schedule_length": len(schedule),
                "schedule_cost": test_eval["schedule_cost"],
                "schedule": schedule_str,
                "maximize": maximize,
                "scenario_budget": budget,
            }
        )

    if os.path.exists(path := output):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark presolvers on ASlib scenarios"
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
        "--presolver-budgets",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0, 50.0, 100.0],
        help="List of presolver time budgets (e.g., --presolver-budgets 10 20 30 50)",
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
        default="/home/ni574034/asf/bench/results/presolver_benchmark.csv",
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
        slurm_job_name="PRESOLVER_BENCH",
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

            for presolver_budget in args.presolver_budgets:
                logger.info(f" presolver_budget={presolver_budget}")
                for fold in range(1, 11):
                    try:
                        executor.submit(
                            run_fold_benchmark,
                            scenario_path=scenario_path,
                            fold=fold,
                            presolver_budget=presolver_budget,
                            par_factor=args.par_factor,
                            output=args.output,
                        )
                    except Exception as e:
                        logger.error(f"Error on fold {fold}: {e}")
