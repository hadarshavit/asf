import asf.scenario.aslib_reader as aslib_reader
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors import (
    JointRanking,
)
from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor
from functools import partial
from asf.metrics.baselines import running_time_closed_gap
import logging

logging.basicConfig(level=logging.DEBUG)


def train_selector(scenario_name: str, selector_class: AbstractSelector) -> None:
    """
    Train a selector using 10-fold cross-validation on the given scenario.

    Parameters:
    scenario_name (str): The name of the scenario to load.
    selector_class (Type[AbstractSelector]): The class of the selector to be trained.

    Returns:
    None
    """
    # Load the scenario
    features, performance, cv, feature_groups, maximize, budget = (
        aslib_reader.read_aslib_scenario(scenario_name)
    )

    performance[performance >= budget] = budget * 10
    solvers_performance = performance.sum(axis=0).sort_values(ascending=True)
    best_solvers = solvers_performance.index[:5]
    algorithms = best_solvers
    performance = performance[algorithms]

    # Initialize the selector
    selector = selector_class(budget=budget, maximize=maximize)

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

        score = running_time_closed_gap(schedules, y_test, budget, par)
        total_score += score
        print(f"Fold score: {score}")

    return running_time_closed_gap(all_schedules, performance, budget, par)


if __name__ == "__main__":
    print(
        train_selector(
            "bench/aslib_data/SAT12-INDU",
            partial(
                JointRanking,
            ),
        )
    )

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
