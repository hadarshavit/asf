from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold

from asf.metrics.baselines import running_time_selector_performance
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.utils.configurable import convert_class_choices_to_categorical
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle

try:
    from ConfigSpace import Configuration

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

try:
    from smac import HyperparameterOptimizationFacade, Scenario

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False


def _create_pipeline(
    config: Configuration | dict[str, Any],
    budget: float | None,
    maximize: bool,
    selector_kwargs: dict[str, Any],
    feature_groups: dict[str, Any] | None,
    max_feature_time: float | None = None,
) -> SelectorPipeline:
    """
    Helper function to create a SelectorPipeline from a configuration.
    """
    pipeline_partial = SelectorPipeline.get_from_configuration(
        configuration=config,
        feature_groups=feature_groups,
        max_feature_time=max_feature_time,
        budget=budget,
        maximize=maximize,
        **selector_kwargs,
    )
    return pipeline_partial()


def tune_selector(
    X: pd.DataFrame,
    y: pd.DataFrame,
    selector_class: type[AbstractSelector]
    | list[type[AbstractSelector]]
    | list[tuple[type[AbstractSelector], dict[str, Any]]],
    features_running_time: pd.DataFrame,
    algorithm_features: pd.DataFrame | None = None,
    selector_kwargs: dict[str, Any] | None = None,
    preprocessing_class: list[type[TransformerMixin]] | None = None,
    pre_solving_class: list[type[Any]] | None = None,
    feature_selector: Any | None = None,
    algorithm_pre_selector: Any | None = None,
    max_algorithm_pre_selector: int | None = None,
    budget: float | None = None,
    maximize: bool = False,
    feature_groups: dict[str, Any] | None = None,
    output_dir: str = "./smac_output",
    smac_metric: Callable[
        ..., float | dict[str, float]
    ] = running_time_selector_performance,
    smac_kwargs: Callable[[Scenario], dict[str, Any]] | None = None,
    smac_scenario_kwargs: dict[str, Any] | None = None,
    runcount_limit: int = 100,
    timeout: float = float("inf"),
    seed: int = 0,
    cv: int = 10,
    groups: np.ndarray | None = None,
    max_feature_time: float | None = None,
) -> SelectorPipeline:
    """
    Tunes a selector model using SMAC.

    Parameters
    ----------
    X : pd.DataFrame
        Instance feature matrix.
    y : pd.DataFrame
        Algorithm performance matrix.
    selector_class : type or list
        Selector classes to tune.
    features_running_time : pd.DataFrame
        Running times for computing feature groups.
    algorithm_features : pd.DataFrame or None, optional
        Features for algorithms.
    selector_kwargs : dict or None, optional
        Arguments for selector instantiation.
    preprocessing_class : list or None, optional
        List of preprocessor classes.
    pre_solving_class : list or None, optional
        List of presolver classes.
    feature_selector : Any or None, optional
        Feature selection component.
    algorithm_pre_selector : Any or None, optional
        Algorithm pre-selection component.
    max_algorithm_pre_selector : int or None, optional
        Constraint for pre-selection.
    budget : float or None, optional
        Global cutoff time.
    maximize : bool, default=False
        Whether to maximize the performance metric.
    feature_groups : dict or None, optional
        Definition of feature groups.
    output_dir : str, default="./smac_output"
        SMAC output directory.
    smac_metric : callable, default=running_time_selector_performance
        Evaluation metric for SMAC.
    smac_kwargs : callable or None, optional
        Additional arguments for SMAC facade.
    smac_scenario_kwargs : dict or None, optional
        Additional arguments for SMAC scenario.
    runcount_limit : int, default=100
        Limit for trials.
    timeout : float, default=inf
        Wall-clock time limit.
    seed : int, default=0
        Random seed.
    cv : int, default=10
        Number of cross-validation folds.
    groups : np.ndarray or None, optional
        Group labels for CV.
    max_feature_time : float or None, optional
        Budget per feature group.

    Returns
    -------
    SelectorPipeline
        Best pipeline found by SMAC.
    """
    _logger = logging.getLogger(__name__)

    if not SMAC_AVAILABLE:
        raise RuntimeError("SMAC is not installed.")
    if not CONFIGSPACE_AVAILABLE:
        raise RuntimeError("ConfigSpace is not installed.")

    if pre_solving_class is not None and budget is None:
        raise ValueError("Budget must be provided if using pre-solving.")

    sel_list = selector_class if isinstance(selector_class, list) else [selector_class]
    sel_kwargs = selector_kwargs or {}
    sc_kwargs = smac_scenario_kwargs or {}

    cs = SelectorPipeline.get_configuration_space(
        selector_class=sel_list,
        preprocessing_class=preprocessing_class,
        pre_solving_class=pre_solving_class,
        feature_groups=feature_groups,
        algorithm_pre_selector=algorithm_pre_selector,
        max_feature_time=max_feature_time,
        budget=budget,
        max_algorithm_pre_selector=max_algorithm_pre_selector,
        n_algorithms=y.shape[1] if hasattr(y, "shape") else None,
        **sel_kwargs,
    )

    cs = convert_class_choices_to_categorical(cs)

    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=Path(output_dir),
        seed=seed,
        **sc_kwargs,
    )

    def target_function(config: Configuration, seed: int) -> float:
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        scores = []
        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            rt_test = features_running_time.iloc[test_idx]

            pipeline = _create_pipeline(
                config,
                budget,
                maximize,
                sel_kwargs,
                feature_groups,
                max_feature_time=max_feature_time,
            )

            pipeline.fit(X_train, y_train, algorithm_features=algorithm_features)
            y_pred = pipeline.predict(X_test)
            assert isinstance(y_pred, dict)  # Added assertion for y_pred type

            score = smac_metric(y_pred, y_test, budget, rt_test)
            if isinstance(score, dict):
                score = float(np.mean(list(score.values())))
            scores.append(float(score))

        final_score = float(np.mean(scores))
        return -final_score if maximize else final_score

    facade_kwargs = smac_kwargs(scenario) if smac_kwargs is not None else {}
    smac = HyperparameterOptimizationFacade(scenario, target_function, **facade_kwargs)
    best_config = smac.optimize()

    if isinstance(best_config, list):
        best_config = best_config[0]
    return _create_pipeline(
        best_config,
        budget,
        maximize,
        sel_kwargs,
        feature_groups,
        max_feature_time=max_feature_time,
    )
