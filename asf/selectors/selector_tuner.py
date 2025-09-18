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

import numpy as np
import pandas as pd

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        UniformFloatHyperparameter,
    )
    from smac import HyperparameterOptimizationFacade, Scenario

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from sklearn.model_selection import KFold


from asf.metrics.baselines import running_time_selector_performance
from sklearn.base import TransformerMixin
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.selector_pipeline import SelectorPipeline
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle


def tune_selector(
    X: pd.DataFrame,
    y: pd.DataFrame,
    selector_class: list[AbstractSelector]
    | AbstractSelector
    | list[tuple[AbstractSelector, dict]],
    selector_kwargs: dict = {},
    preprocessing_class: list[TransformerMixin] = None,
    pre_solving_class: list[object] = None,
    feature_selector: object = None,
    algorithm_pre_selector: object = None,
    budget: float = None,
    maximize: bool = False,
    feature_groups: list = None,
    output_dir: str = "./smac_output",
    smac_metric: callable = running_time_selector_performance,
    smac_kwargs: dict = {},
    smac_scenario_kwargs: dict = {},
    runcount_limit: int = 100,
    timeout: float = np.inf,
    seed: int = 0,
    cv: int = 10,
    groups: np.ndarray = None,
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
        feature_groups (list, optional): Feature groups to consider. Defaults to None.
        output_dir (str): Directory to store SMAC output. Defaults to "./smac_output".
        smac_metric (callable): Metric function to evaluate the selector's performance. Defaults to `running_time_selector_performance`.
        smac_kwargs (dict): Additional arguments for SMAC's optimization facade.
        smac_scenario_kwargs (dict): Additional arguments for SMAC's scenario configuration.
        runcount_limit (int): Maximum number of function evaluations. Defaults to 100.
        timeout (float): Maximum wall-clock time for optimization. Defaults to np.inf.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        cv (int): Number of cross-validation splits. Defaults to 10.
        groups (np.ndarray, optional): Group labels for cross-validation. Defaults to None.

    Returns:
        SelectorPipeline: A pipeline with the best-tuned selector and preprocessing steps.
    """
    assert CONFIGSPACE_AVAILABLE, (
        "SMAC is not installed. Please install it to use this function via pip install asf-lib[tune]."
    )

    if pre_solving_class is not None and len(pre_solving_class) > 0 and budget is None:
        raise ValueError(
            "If pre_solving_class is provided, you must also provide a budget."
        )

    if type(selector_class) is not list:
        selector_class = [selector_class]

    cs = ConfigurationSpace()
    cs_transform = {}

    # Add selectors to configuration space
    if type(selector_class[0]) is tuple:
        selector_param = Categorical(
            name="selector",
            items=[str(c[0].__name__) for c in selector_class],
        )
        cs_transform["selector"] = {str(c[0].__name__): c[0] for c in selector_class}
    else:
        selector_param = Categorical(
            name="selector",
            items=[str(c.__name__) for c in selector_class],
        )
        cs_transform["selector"] = {str(c.__name__): c for c in selector_class}
    cs.add(selector_param)

    for selector in selector_class:
        if type(selector) is tuple:
            selector_space_kwargs = selector[1]
            selector = selector[0]
        else:
            selector_space_kwargs = {}

        cs, cs_transform = selector.get_configuration_space(
            cs=cs,
            cs_transform=cs_transform,
            parent_param=selector_param,
            parent_value=str(selector.__name__),
            **selector_space_kwargs,
        )

    # Add pre-solving and budget to configuration space
    if pre_solving_class is not None and len(pre_solving_class) > 0:
        presolver_param = Categorical(
            name="presolver",
            items=[str(type(p).__name__) for p in pre_solving_class],
        )
        cs_transform["presolver"] = {
            str(type(p).__name__): p for p in pre_solving_class
        }
        cs.add(presolver_param)
        # Budget for presolver (fraction of total budget)
        presolver_budget_param = UniformFloatHyperparameter(
            name="presolver_budget",
            lower=0.0,
            upper=1.0,
            default_value=0.2,
        )
        cs.add(presolver_budget_param)

    # Add preprocessors to configuration spaces
    if preprocessing_class is not None and len(preprocessing_class) > 0:
        # Use a multi-categorical: for each preprocessor, a boolean flag
        for i, preproc in enumerate(preprocessing_class):
            preproc_param = Categorical(
                name=f"preprocessor_{i}",
                items=["off", "on"],
            )
            cs.add(preproc_param)
        cs_transform["preprocessors"] = preprocessing_class

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

            # Preprocessor selection
            preprocessors = None
            if "preprocessors" in cs_transform:
                preprocessors = []
                for i, preproc in enumerate(cs_transform["preprocessors"]):
                    if config.get(f"preprocessor_{i}", "off") == "on":
                        preprocessors.append(preproc)
                if len(preprocessors) == 0:
                    preprocessors = None

            # Presolver selection and budget
            presolver = None
            presolver_budget = None
            if "presolver" in cs_transform:
                presolver = cs_transform["presolver"][config["presolver"]]
                presolver_budget = (
                    config["presolver_budget"] * budget if budget is not None else None
                )
                if presolver is not None and presolver_budget is not None:
                    setattr(presolver, "budget", presolver_budget)

            selector = SelectorPipeline(
                selector=cs_transform["selector"][
                    config["selector"]
                ].get_from_configuration(
                    config,
                    cs_transform,
                    budget=(budget - presolver_budget)
                    if presolver_budget is not None
                    else budget,
                    maximize=maximize,
                    feature_groups=feature_groups,
                    **selector_kwargs,
                ),
                preprocessor=preprocessors,
                pre_solving=presolver,
                feature_selector=feature_selector,
                algorithm_pre_selector=algorithm_pre_selector,
                budget=budget,
                maximize=maximize,
                feature_groups=feature_groups,
            )
            selector.fit(X_train, y_train)

            y_pred = selector.predict(X_test)
            score = smac_metric(y_pred, y_test)
            scores.append(score)

        return np.mean(scores)

    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    del smac  # clean up SMAC to free memory and delete dask client

    # Final pipeline construction
    preprocessors = None
    if "preprocessors" in cs_transform:
        preprocessors = []
        for i, preproc in enumerate(cs_transform["preprocessors"]):
            if best_config.get(f"preprocessor_{i}", "off") == "on":
                preprocessors.append(preproc)
        if len(preprocessors) == 0:
            preprocessors = None

    presolver = None
    presolver_budget = None
    if "presolver" in cs_transform:
        presolver = cs_transform["presolver"][best_config["presolver"]]
        presolver_budget = (
            best_config["presolver_budget"] * budget if budget is not None else None
        )
        setattr(presolver, "budget", presolver_budget)

    return SelectorPipeline(
        selector=cs_transform["selector"][
            best_config["selector"]
        ].get_from_configuration(
            best_config,
            cs_transform,
            budget=(budget - presolver_budget)
            if presolver_budget is not None
            else budget,
            maximize=maximize,
            feature_groups=feature_groups,
            **selector_kwargs,
        ),
        preprocessor=preprocessors,
        pre_solving=presolver,
        feature_selector=feature_selector,
        algorithm_pre_selector=algorithm_pre_selector,
        budget=budget,
        maximize=maximize,
        feature_groups=feature_groups,
    )
