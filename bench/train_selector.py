import asf.scenario.aslib_reader as aslib_reader
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors import (
    JointRanking,
)
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor
from functools import partial
from asf.metrics.baselines import running_time_closed_gap
import logging

from asf.selectors import PairwiseClassifier, MultiClassClassifier, PerformanceModel
from sklearn.ensemble import RandomForestClassifier

from asf.pre_selector.knee_of_the_curve_pre_selector import KneeOfCurvePreSelector
from asf.pre_selector.marginal_contribution_based import MarginalContributionBasedPreSelector
from asf.metrics.baselines import virtual_best_solver
from asf.selectors.selector_pipeline import SelectorPipeline

logging.basicConfig(level=logging.DEBUG)


def train_selector(scenario_name: str, selector_class: AbstractSelector, algorithm_pre_selector=None) -> None:
    """
    Train a selector using 10-fold cross-validation on the given scenario.

    Parameters:
    scenario_name (str): The name of the scenario to load.
    selector_class (Type[AbstractSelector]): The class of the selector to be trained.

    Returns:
    None
    """
    # Load the scenario
    features, performance, features_running_time, cv, feature_groups, maximize, budget = (
        aslib_reader.read_aslib_scenario(scenario_name)
    )

    print(cv)
    input()

    performance[performance >= budget] = budget * 10
    solvers_performance = performance.sum(axis=0).sort_values(ascending=True)
    best_solvers = solvers_performance.index[:5]
    algorithms = best_solvers
    performance = performance[algorithms]

    # Initialize the selector pipeline with the pre-selector
    selector = SelectorPipeline(
        selector=selector_class(budget=budget, maximize=maximize),
        algorithm_pre_selector=algorithm_pre_selector,
        preprocessor=get_default_preprocessor(),
        budget=budget,
        maximize=maximize,
        feature_groups=feature_groups,
    )

    total_score = 0

    all_schedules = {}
    par = 10
    # Perform 10-fold cross-validation
    for i in range(10):
        X_train, X_test = features[cv["fold"] != i + 1], features[cv["fold"] == i + 1]
        y_train, y_test = (
            performance[cv["fold"] != i + 1],
            performance[cv["fold"] == i + 1],
        )

        preprocessor = get_default_preprocessor()
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Train the selector
        selector.fit(X_train, y_train)
        # selector.fit(X_test, y_test)
        schedules = selector.predict(X_test)
        all_schedules.update(schedules)
        # Evaluate the selector

        score = running_time_closed_gap(schedules, y_test, budget, par, features_running_time)
        total_score += score
        print(f"Fold score: {score}")

    return running_time_closed_gap(all_schedules, performance, budget, par, features_running_time)


if __name__ == "__main__":
    knee_preselector = KneeOfCurvePreSelector(
        metric=virtual_best_solver,
        base_pre_selector=MarginalContributionBasedPreSelector,
        maximize=False,  # or True, depending on your scenario
        S=1.0,
        workers=1,
    )

    print(
        train_selector(
            "/home/schiller/asf/aslib_data/SAT12-INDU",
            partial(PairwiseClassifier, model_class=RandomForestClassifier),
            knee_preselector,
        )
    )

    print(
        train_selector(
            "/home/schiller/asf/aslib_data/SAT12-INDU",
            partial(MultiClassClassifier, model_class=RandomForestClassifier),
            knee_preselector,
        )
    )

    # print(
    #     train_selector(
    #         "/home/schiller/asf/aslib_data/SAT12-INDU",
    #         partial(PerformanceModel, use_multi_target=False, model_class=RandomForestClassifier),
    #         knee_preselector,
    #     )
    # )


    # print(
    #     train_selector(
    #         "bench/aslib_data/SAT12-INDU",
    #         partial(
    #             JointRanking,
    #         ),
    #     )
    # )

    # print(
    #     train_selector(
    #         "bench/aslib_data/SAT12-INDU",
    #         partial(
    #             PerformanceModel,
    #             use_multi_target=False,
    #         ),
    #         RegressionMLP,
    #     )
    # )
    # print(
    #     train_selector(
    #         "bench/aslib_data/SAT12-INDU",
    #         SimpleRanking,
    #         partial(
    #             SklearnWrapper,
    #             XGBRanker,
    #             init_params={
    #                 "objective": "rank:pairwise",
    #                 # "lambdarank_pair_method": "topk",
    #                 # "lambdarank_num_pair_per_sample": 10,
    #             },
    #         ),
    #     )
    # )
    # print(
    #     train_selector(
    #         "bench/aslib_data/SAT12-INDU", PairwiseClassifier, RandomForestClassifier
    #     )
    # )
    # print(
    #     train_selector(
    #         "bench/aslib_data/SAT12-INDU",
    #         PairwiseRegressor,
    #         partial(
    #             SklearnWrapper,
    #             RandomForestRegressor,
    #             init_params={"n_estimators": 100, "max_features": "sqrt"},
    #         ),
    #     )
    # )
