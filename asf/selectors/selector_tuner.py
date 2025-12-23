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
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        UniformFloatHyperparameter,
        ForbiddenAndConjunction,
        ForbiddenEqualsClause,
    )

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
from asf.preprocessing.feature_group_selector import FeatureGroupSelector


def _create_pipeline(
    config,
    cs_transform,
    budget,
    maximize,
    selector_kwargs,
    feature_selector,
    algorithm_pre_selector,
    max_feature_time: float | None = None,
):
    """Helper function to create a SelectorPipeline from a configuration.

    Parameters:
        max_feature_time: Optional budget (seconds) to allocate per feature group.
    """
    # Preprocessor selection
    preprocessors = None
    if "preprocessors" in cs_transform:
        preprocessors = [
            preproc
            for i, preproc in enumerate(cs_transform["preprocessors"])
            if config.get(f"preprocessor_{i}")
        ]
        if not preprocessors:
            preprocessors = None

    # Presolver selection and budget
    presolver = None
    presolver_budget = None
    if "presolver" in cs_transform:
        if config["presolver"] != "None":
            presolver_class = cs_transform["presolver"][config["presolver"]]
            presolver_name = config["presolver"]
            # Extract presolver_budget from configuration
            budget_key = f"{presolver_name}:presolver_budget"
            if budget_key in config:
                presolver_budget = config[budget_key]

            if presolver_class is not None:
                presolver = presolver_class.get_from_configuration(
                    configuration=config,
                    cs_transform=cs_transform,
                    budget=presolver_budget,
                    maximize=maximize,
                    presolver_name=presolver_name,
                )

    # Feature group selection
    selected_feature_groups = None
    feature_groups_info = None
    if "feature_groups" in cs_transform:
        feature_groups_info = cs_transform["feature_groups"]
        selected_feature_groups = FeatureGroupSelector.get_selected_groups_from_config(
            feature_groups_info, config, prefix="feature_group:"
        )

    # Algorithm pre-selector configuration
    current_algorithm_pre_selector = None
    if "algorithm_pre_selector" in cs_transform:
        pre_selector_name = config["algorithm_pre_selector"]
        pre_selector_class = cs_transform["algorithm_pre_selector"][pre_selector_name]
        pre_selector_defaults = cs_transform.get("algorithm_pre_selector_defaults", {})

        if pre_selector_class is not None:
            current_algorithm_pre_selector = pre_selector_class.get_from_configuration(
                configuration=config,
                cs_transform=cs_transform,
                maximize=maximize,
                pre_selector_name=pre_selector_name,
                **pre_selector_defaults,  # Pass default kwargs
            )

    selector_instance = cs_transform["selector"][
        config["selector"]
    ].get_from_configuration(
        config,
        cs_transform,
        budget=budget,  # Don't subtract presolver budget - matches AutoFolio behavior
        maximize=maximize,
        feature_groups=selected_feature_groups,
        **selector_kwargs,
    )

    # Determine the effective max_feature_time: prefer value from config if present
    effective_max_feature_time = max_feature_time
    if "max_feature_time" in config:
        # SMAC stores floats directly in the configuration
        try:
            effective_max_feature_time = float(config["max_feature_time"])
        except Exception:
            effective_max_feature_time = effective_max_feature_time

    # Create pipeline and store max feature time (cap) on it for later evaluation
    pipeline = SelectorPipeline(
        selector=selector_instance,
        preprocessor=preprocessors,
        pre_solving=presolver,
        feature_selector=feature_selector,
        algorithm_pre_selector=current_algorithm_pre_selector,
        feature_groups=selected_feature_groups,
        max_feature_time=effective_max_feature_time,
    )

    return pipeline


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
    if pre_solving_class is not None:
        if type(pre_solving_class) is not list:
            pre_solving_class = [pre_solving_class]

        # Create presolver selection parameter
        presolver_param = Categorical(
            name="presolver",
            items=[p.__name__ for p in pre_solving_class] + ["None"],
        )
        cs_transform["presolver"] = {p.__name__: p for p in pre_solving_class}
        cs.add(presolver_param)

        # Add each presolver's configuration space (including presolver_budget)
        for presolver_class in pre_solving_class:
            cs, cs_transform = presolver_class.get_configuration_space(
                cs=cs,
                cs_transform=cs_transform,
                parent_param=presolver_param,
                parent_value=presolver_class.__name__,
                total_budget=budget,
            )

    # Add preprocessors to configuration spaces
    if preprocessing_class is not None and len(preprocessing_class) > 0:
        # Use a multi-categorical: for each preprocessor, a boolean flag
        for i, preproc in enumerate(preprocessing_class):
            preproc_param = Categorical(
                name=f"preprocessor_{i}",
                items=[True, False],
            )
            cs.add(preproc_param)
        cs_transform["preprocessors"] = preprocessing_class

    # Add feature groups to configuration space (each can be enabled/disabled)
    # Also add forbidden clauses for prerequisite groups
    fg_params = {}  # Store params to create forbidden clauses
    if feature_groups is not None and len(feature_groups) > 0:
        # If only one feature group, don't add it as a hyperparameter (it must always be True)
        # Only add hyperparameters if there are multiple feature groups
        if len(feature_groups) > 1:
            # First, add all feature group parameters
            for fg_name in feature_groups.keys():
                fg_param = Categorical(
                    name=f"feature_group:{fg_name}",
                    items=[True, False],
                    default=True,
                )
                cs.add(fg_param)
                fg_params[fg_name] = fg_param

            # Then, add forbidden clauses for prerequisite requirements
            # If a group requires another, forbid: group=True AND required=False
            for fg_name, fg_info in feature_groups.items():
                required_groups = fg_info.get("requires", [])
                for required_group in required_groups:
                    if required_group in fg_params:
                        # Forbid: fg_name=True AND required_group=False
                        forbidden = ForbiddenAndConjunction(
                            ForbiddenEqualsClause(fg_params[fg_name], True),
                            ForbiddenEqualsClause(fg_params[required_group], False),
                        )
                        cs.add(forbidden)

            # CRITICAL: Forbid having ALL feature groups disabled
            # At least one feature group must be enabled, otherwise the selector has no features!
            # Create forbidden clause: forbid all groups being False simultaneously
            all_false_clauses = [
                ForbiddenEqualsClause(param, False) for param in fg_params.values()
            ]
            forbidden_all_false = ForbiddenAndConjunction(*all_false_clauses)
            cs.add(forbidden_all_false)

        cs_transform["feature_groups"] = feature_groups

    if algorithm_pre_selector is not None:
        # Extract class and default kwargs if tuple format (similar to selector_class)
        if isinstance(algorithm_pre_selector, tuple):
            pre_selector_class = algorithm_pre_selector[0]
            pre_selector_defaults = algorithm_pre_selector[1]
        else:
            pre_selector_class = algorithm_pre_selector
            pre_selector_defaults = {}

        # Create algorithm pre-selector selection parameter
        pre_selector_param = Categorical(
            name="algorithm_pre_selector",
            items=[str(pre_selector_class.__name__)],
        )
        cs_transform["algorithm_pre_selector"] = {
            str(pre_selector_class.__name__): pre_selector_class
        }
        # Store default kwargs for later instantiation
        cs_transform["algorithm_pre_selector_defaults"] = pre_selector_defaults
        cs.add(pre_selector_param)

        # Add the algorithm pre-selector's configuration space (including n_algorithms)
        cs, cs_transform = pre_selector_class.get_configuration_space(
            cs=cs,
            cs_transform=cs_transform,
            parent_param=pre_selector_param,
            parent_value=str(pre_selector_class.__name__),
            n_algorithms_max=y.shape[1]
            if max_algorithm_pre_selector is None
            else max_algorithm_pre_selector,
        )

    # Add max feature computation time hyperparameter to the config space if the
    # user did not pass a fixed cap. This allows SMAC to tune a cap (seconds)
    # on the total per-instance feature computation time used during HPO.
    if max_feature_time:
        # upper bound: use budget as a safe ceiling if available, otherwise 3600s
        upper = float(budget) if budget is not None else 3600.0
        mf_param = UniformFloatHyperparameter(
            name="max_feature_time",
            lower=0.0,
            upper=upper,
            default_value=min(60.0, upper),
            log=False,
        )
        cs.add(mf_param)

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
                cs_transform,
                budget,
                maximize,
                selector_kwargs,
                feature_selector,
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
        cs_transform,
        budget,
        maximize,
        selector_kwargs,
        feature_selector,
        algorithm_pre_selector,
        max_feature_time=max_feature_time,
    )
