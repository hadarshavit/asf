import logging
import os
from functools import partial
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
)
from asf.presolving import Aspeed
from asf.selectors.selector_tuner import tune_selector

# Configure logging to print debug logs for all loggers
logging.basicConfig(level=logging.DEBUG, force=True)


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

    score, result_selector = evaluate_selector(
        selector_class=selector,
        scenario_path=os.path.join(base_path, scenario),
        fold=fold,
        # hpo_func=None,
        hpo_func=tune_selector,
        hpo_kwargs={
            "runcount_limit": 100,
            "cv": 10,
            "smac_kwargs": lambda scenario: {"overwrite": True, "logging_level": False},
            "output_dir": f"/home/shavit/asf/bench/results/{scenario}_{selector_name}_fold{fold}_selector_tuning",
            "max_algorithm_pre_selector": 20,
            "pre_solving_class": Aspeed if scenario != "OPENML-WEKA-2017" else None,
        },
        algorithm_pre_selector=partial(
            MarginalContributionBasedPreSelector,
            metric=virtual_best_solver,
            mode="forward",
        ),
    )

    print(f"Selector: {result_selector} Test score: {score}")

    results = [
        {
            "scenario": scenario,
            "fold": fold,
            "selector": selector_name,
            "test_score": score,
        }
    ]

    if os.path.exists(path := f"/home/shavit/asf/bench/results/results_{scenario}.csv"):
        pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
    else:
        pd.DataFrame(results).to_csv(path, header=True, index=False)


if __name__ == "__main__":
    selectors = [
        (MultiClassClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (PairwiseClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (PairwiseRegressor, {"model_class": [RandomForestRegressorWrapper]}),
        (PerformanceModel, {"model_class": [RandomForestRegressorWrapper]}),
    ]
    # selectors = [
    #     PairwiseClassifier,
    #     MultiClassClassifier,
    #     PerformanceModel,
    # ]

    scenarios = [
        "SAT12-INDU",
        "SAT20-MAIN",
        "TSP-LION2015",
        "QBF-2016",
        "MIP-2016",
        "MAXSAT19-UCMS",
        "CSP-Minizinc-Time-2016",
        "ASP-POTASSCO",
        "OPENML-WEKA-2017",
        "BNSL-2016",
        "GRAPHS-2015",
    ]

    for selector in selectors:
        for scenario in scenarios:
            for fold in range(1, 11):
                run(
                    selector,
                    scenario,
                    fold,
                    base_path="/home/shavit/asf/paper/aslib_data",
                )
