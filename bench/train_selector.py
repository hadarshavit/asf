from asf.pre_selector.knee_of_the_curve_pre_selector import KneeOfCurvePreSelector
from asf.pre_selector.marginal_contribution_based import (
    MarginalContributionBasedPreSelector,
)
from asf.scenario.aslib_reader import evaluate_selector
from asf.metrics.baselines import virtual_best_solver
from asf.selectors import (
    PairwiseClassifier,
    MultiClassClassifier,
    PerformanceModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from functools import partial


if __name__ == "__main__":
    knee_preselector = KneeOfCurvePreSelector(
        metric=virtual_best_solver,
        base_pre_selector=MarginalContributionBasedPreSelector,
        maximize=False,
        S=1.0,
        workers=1,
    )

    selectors = [
        PairwiseClassifier(model_class=RandomForestClassifier),
        MultiClassClassifier(model_class=RandomForestClassifier),
        partial(PerformanceModel, model_class=RandomForestRegressor),
    ]

    scenarios = [
        "/home/schiller/asf/aslib_data/SAT12-INDU",
        "/home/schiller/asf/aslib_data/SAT20-MAIN",
        "/home/schiller/asf/aslib_data/TSP-LION2015",
        "/home/schiller/asf/aslib_data/QBF-2016",
        "/home/schiller/asf/aslib_data/MIP-2016",
        "/home/schiller/asf/aslib_data/MAXSAT19-UCMS",
        "/home/schiller/asf/aslib_data/IPC2018",
        "/home/schiller/asf/aslib_data/CSP-Minizinc-Time-2016",
        "/home/schiller/asf/aslib_data/ASP-POTASSCO",
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
                # hpo_func=tune_selector,
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
