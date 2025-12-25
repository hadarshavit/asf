"""
This module provides functionality for tuning selector models using SMAC (Sequential Model-based Algorithm Configuration).
The `tune_selector` function optimizes hyperparameters for selector models, allowing for flexible configuration
of preprocessing, feature selection, and algorithm selection pipelines.

Dependencies:
- numpy
- pandas
- ConfigSpace
- sklearn
- smac
- asf (custom modules)
"""

import logging
import time
import numpy as np
import pandas as pd

try:
    import ConfigSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

try:
    from smac import HyperparameterOptimizationFacade, Scenario

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
from sklearn.model_selection import KFold


from asf.metrics.baselines import running_time_selector_performance
from sklearn.base import TransformerMixin
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle
from asf.utils.configurable import convert_class_choices_to_categorical


def _create_pipeline(
    config,
    budget,
    maximize,
    selector_kwargs,
    selector_class,
    preprocessing_class,
    pre_solving_class,
    feature_groups,
    algorithm_pre_selector,
    max_feature_time: float | None = None,
):
    """Helper function to create a SelectorPipeline from a configuration."""

    # Use SelectorPipeline.get_from_configuration directly
    pipeline_partial = SelectorPipeline.get_from_configuration(
        configuration=config,
        selector_class=selector_class,
        preprocessing_class=preprocessing_class,
        pre_solving_class=pre_solving_class,
        feature_groups=feature_groups,
        algorithm_pre_selector=algorithm_pre_selector,
        max_feature_time=max_feature_time,
        budget=budget,
        maximize=maximize,
        **selector_kwargs,  # passed to selector
    )

    return pipeline_partial()


def tune_selector(
    X: pd.DataFrame,
    y: pd.DataFrame,
    selector_class: list[AbstractSelector]
    | AbstractSelector
    | list[tuple[AbstractSelector, dict]],
    features_running_time: pd.DataFrame,
    algorithm_features=None,
    selector_kwargs: dict = {},
    preprocessing_class: list[TransformerMixin] = None,
    pre_solving_class: list[object] = None,
    feature_selector: object = None,
    algorithm_pre_selector: object = None,
    max_algorithm_pre_selector: int = None,
    budget: float = None,
    maximize: bool = False,
    feature_groups: dict = None,
    output_dir: str = "./smac_output",
    smac_metric: callable = running_time_selector_performance,
    smac_kwargs: callable = None,
    smac_scenario_kwargs: dict = {},
    runcount_limit: int = 100,
    timeout: float = np.inf,
    seed: int = 0,
    cv: int = 10,
    groups: np.ndarray = None,
    max_feature_time: float | None = False,
) -> SelectorPipeline:
    """
    Tunes a selector model using SMAC for hyperparameter optimization.

    Parameters:
        X (pd.DataFrame): Feature matrix for training and testing.
        y (pd.DataFrame): Target matrix for training and testing.
        selector_class (list[AbstractSelector]): List of selector classes to tune. Defaults to [PairwiseClassifier, PairwiseRegressor].
        selector_space_kwargs (dict): Additional arguments for the selector's configuration space.
        selector_kwargs (dict): Additional arguments for the selector's instantiation.
        preprocessing_class (AbstractPreprocessor, optional): Preprocessing class to apply before selector. Defaults to None.
        pre_solving_class (object, optional): Pre-solving strategies to use. Defaults to None.
        feature_selector (object, optional): Feature selector to use. Defaults to None.
        algorithm_pre_selector (object, optional): Algorithm pre-selector to use. Defaults to None.
        budget (float, optional): Budget for the selector. Defaults to None.
        maximize (bool): Whether to maximize the metric. Defaults to False.
        feature_groups (dict, optional): Feature groups to consider. Each key is a feature group name,
            and the value is a dict with 'provides' key listing feature names in that group.
            When provided, SMAC will optimize which feature groups to use by adding a boolean
            hyperparameter for each group. Only features from selected groups will be used.
            Defaults to None.
        output_dir (str): Directory to store SMAC output. Defaults to "./smac_output".
        smac_metric (callable): Metric function to evaluate the selector's performance. Defaults to `running_time_selector_performance`.
        smac_kwargs (callable): Additional arguments for SMAC's optimization facade.
        smac_scenario_kwargs (dict): Additional arguments for SMAC's scenario configuration.
        runcount_limit (int): Maximum number of function evaluations. Defaults to 100.
        timeout (float): Maximum wall-clock time for optimization. Defaults to np.inf.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        cv (int): Number of cross-validation splits. Defaults to 10.
        groups (np.ndarray, optional): Group labels for cross-validation. Defaults to None.
        max_feature_time (float, optional): A budget (in seconds) to allocate per feature group.
            When set, each feature group in the schedule will be given this time budget. The metric
            will use min(actual_time, budget) for each feature group. Defaults to None.

    Returns:
        SelectorPipeline: A pipeline with the best-tuned selector and preprocessing steps.
    """
    _logger = logging.getLogger(__name__)

    if not SMAC_AVAILABLE:
        raise RuntimeError("SMAC is not installed. Install it with: pip install smac")
    if not CONFIGSPACE_AVAILABLE:
        raise RuntimeError(
            "ConfigSpace is not installed. Install it with: pip install ConfigSpace"
        )

    if pre_solving_class is not None and budget is None:
        raise ValueError(
            "If pre_solving_class is provided, you must also provide a budget."
        )

    if type(selector_class) is not list:
        selector_class = [selector_class]

    cs = SelectorPipeline.get_configuration_space(
        selector_class=selector_class,
        preprocessing_class=preprocessing_class,
        pre_solving_class=pre_solving_class,
        feature_groups=feature_groups,
        algorithm_pre_selector=algorithm_pre_selector,
        max_feature_time=max_feature_time,
        budget=budget,
        max_algorithm_pre_selector=max_algorithm_pre_selector,
        n_algorithms=y.shape[1] if hasattr(y, "shape") else None,
        **selector_kwargs,
    )

    # Convert ClassChoice hyperparameters to regular Categorical for SMAC serialization
    cs = convert_class_choices_to_categorical(cs)

    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    def target_function(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            features_running_time_test = features_running_time.iloc[test_idx]

            pipeline = _create_pipeline(
                config,
                budget,
                maximize,
                selector_kwargs,
                selector_class,
                preprocessing_class,
                pre_solving_class,
                feature_groups,
                algorithm_pre_selector,
                max_feature_time=max_feature_time,
            )

            pipeline.fit(X_train, y_train, algorithm_features=algorithm_features)
            y_pred = pipeline.predict(X_test)

            # max_feature_time is no longer passed to metric; budgets are in the schedule itself
            start = time.time()
            score = smac_metric(y_pred, y_test, budget, features_running_time_test)
            _logger.debug(f"Scoring completed in {time.time() - start:.2f} seconds")

            scores.append(score)

        score = np.mean(scores)

        if maximize:
            return -score
        return score

    smac_kwargs = smac_kwargs(scenario) if smac_kwargs is not None else {}
    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    del smac  # clean up SMAC to free memory and delete dask client

    # Final pipeline construction
    return _create_pipeline(
        best_config,
        budget,
        maximize,
        selector_kwargs,
        selector_class,
        preprocessing_class,
        pre_solving_class,
        feature_groups,
        algorithm_pre_selector,
        max_feature_time=max_feature_time,
    )
