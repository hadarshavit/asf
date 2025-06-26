from asf.pre_selector.knee_of_the_curve_pre_selector import KneeOfCurvePreSelector
from asf.pre_selector import OptimizePreSelection
from asf.scenario.aslib_reader import evaluate_selector
from asf.metrics.baselines import virtual_best_solver
from asf.selectors import (
    PairwiseClassifier,
    MultiClassClassifier,
    PerformanceModel,
)
from asf.selectors.selector_tuner import tune_selector
from asf.predictors import RandomForestClassifierWrapper, RandomForestRegressorWrapper


if __name__ == "__main__":
    knee_preselector = KneeOfCurvePreSelector(
        metric=virtual_best_solver,
        base_pre_selector=OptimizePreSelection,
        maximize=False,
        S=1.0,
        workers=1,
    )

    selectors = [
        (PairwiseClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (MultiClassClassifier, {"model_class": [RandomForestClassifierWrapper]}),
        (PerformanceModel, {"model_class": [RandomForestRegressorWrapper]}),
    ]

    scenarios = [
        "aslib_data/SAT12-INDU",
        "aslib_data/SAT20-MAIN",
        "aslib_data/TSP-LION2015",
        "aslib_data/QBF-2016",
        "aslib_data/MIP-2016",
        "aslib_data/MAXSAT19-UCMS",
        "aslib_data/IPC2018",
        "aslib_data/CSP-Minizinc-Time-2016",
        "asf/aslib_data/ASP-POTASSCO",
    ]

    for selector in selectors:
        # Print name of the selector
        if hasattr(selector, "__name__"):
            selector_name = selector.__name__
        elif hasattr(selector, "func"):  # for functools.partial
            selector_name = selector.func.__name__
        else:
            selector_name = selector.__class__.__name__

        print(f"Evaluating selector: {selector_name}")

        for scenario in scenarios:
            print(f"Evaluating scenario: {scenario}")

            score, result_selector = evaluate_selector(
                selector_class=selector,
                scenario_path=scenario,
                fold=10,
                hpo_func=tune_selector,
                hpo_kwargs={
                    "runcount_limit": 100,
                    "cv": 10,
                    "smac_kwargs": {"overwrite": True},
                    "output_dir": f"smac_output/{scenario.split('/')[-1]}",
                },
                algorithm_pre_selector=knee_preselector,
            )

            print(f"Test score: {score}")
            print(f"Selector: {result_selector}")
