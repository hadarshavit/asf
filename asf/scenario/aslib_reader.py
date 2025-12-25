"""
Utilities for reading and evaluating algorithm selection scenarios in ASlib format.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, overload

import pandas as pd

from asf.metrics.baselines import running_time_closed_gap
from asf.metrics.par10 import apply_par
from asf.selectors.baselines import VirtualBestSolver
from asf.selectors.selector_pipeline import SelectorPipeline

try:
    import yaml
    from arff import load
    from yaml import SafeLoader as Loader

    ASLIB_AVAILABLE = True
except ImportError:
    ASLIB_AVAILABLE = False


def read_aslib_scenario(
    path: str,
    add_running_time_features: bool = True,
    training_par_factor: float | None = 10.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    bool,
    float,
    pd.DataFrame | None,
]:
    """
    Read an ASlib scenario from a directory.

    Parameters
    ----------
    path : str
        The path to the ASlib scenario directory.
    add_running_time_features : bool, default=True
        Whether to include running time features (feature costs).
    training_par_factor : float or None, default=10.0
        PAR factor to apply to training performance data. Timeouts (values > budget)
        are replaced with budget * training_par_factor. Set to None to disable.

    Returns
    -------
    tuple
        A tuple containing (features, performance, features_running_time, cv,
        feature_groups, maximize, budget, algorithm_features) where:
        - features: pd.DataFrame of feature values per instance.
        - performance: pd.DataFrame of algorithm performance per instance.
        - features_running_time: pd.DataFrame of feature costs per instance.
        - cv: pd.DataFrame of cross-validation fold assignments.
        - feature_groups: dict of feature group definitions.
        - maximize: bool, True if higher performance values are better.
        - budget: float, the algorithm cutoff time.
        - algorithm_features: pd.DataFrame of algorithm features, or None.

    Raises
    ------
    ImportError
        If the required libraries (pyyaml, liac-arff) are not available.
    """
    if not ASLIB_AVAILABLE:
        raise ImportError(
            "The aslib reader requires 'pyyaml' and 'liac-arff'. "
            "Install them via 'pip install asf[aslib]'."
        )

    description_path = os.path.join(path, "description.txt")
    performance_path = os.path.join(path, "algorithm_runs.arff")
    features_path = os.path.join(path, "feature_values.arff")
    features_runstatus_path = os.path.join(path, "feature_runstatus.arff")
    features_running_time_path = os.path.join(path, "feature_costs.arff")
    cv_path = os.path.join(path, "cv.arff")
    algorithm_features_path = os.path.join(path, "algorithm_feature_values.arff")
    algorithm_features_runstatus_path = os.path.join(
        path, "algorithm_feature_runstatus.arff"
    )

    # Load description file
    with open(description_path, "r") as f:
        description: dict[str, Any] = yaml.load(f, Loader=Loader)

    feature_groups: dict[str, Any] = description["feature_steps"]
    algorithm_feature_groups: dict[str, Any] = description.get(
        "algorithm_feature_steps", {}
    )
    maximize: bool = description["maximize"]
    if not isinstance(maximize, bool):
        maximize = bool(maximize[0])
    budget: float = description["algorithm_cutoff_time"]

    # Load performance data
    with open(performance_path, "r") as f:
        performance_data: dict[str, Any] = load(f)
    performance = pd.DataFrame(
        performance_data["data"],
        columns=[a[0] for a in performance_data["attributes"]],  # type: ignore[arg-type]
    )

    runtime_col = description["performance_measures"][0]

    if "runstatus" in performance.columns:
        performance.loc[performance["runstatus"] != "ok", runtime_col] = budget + 1

    group_cols = ["instance_id", "algorithm"]
    performance = performance.groupby(group_cols, as_index=False)[runtime_col].mean()

    performance = performance.pivot(
        index="instance_id", columns="algorithm", values=runtime_col
    )

    # Load feature values
    with open(features_path, "r") as f:
        features_data: dict[str, Any] = load(f)
    features = pd.DataFrame(
        features_data["data"],
        columns=[a[0] for a in features_data["attributes"]],  # type: ignore[arg-type]
    )

    if os.path.exists(features_runstatus_path):
        with open(features_runstatus_path, "r") as f:
            feature_runstatus_data: dict[str, Any] = load(f)
        feature_runstatus = pd.DataFrame(
            feature_runstatus_data["data"],
            columns=[a[0] for a in feature_runstatus_data["attributes"]],  # type: ignore[arg-type]
        )
        for step_name, step_info in feature_groups.items():
            if step_name in feature_runstatus.columns:
                failed_mask = feature_runstatus[step_name] != "ok"
                failed_instances = feature_runstatus.loc[
                    failed_mask, "instance_id"
                ].values
                step_features = step_info.get("provides", [])
                for feat in step_features:
                    if feat in features.columns:
                        features.loc[
                            features["instance_id"].isin(failed_instances), feat
                        ] = float("nan")

    features = features.groupby("instance_id").mean()
    if "repetition" in features.columns:
        features = features.drop(columns=["repetition"])

    features_running_time = pd.DataFrame(
        0.0,
        index=performance.index,
        columns=["feature_time"],  # type: ignore[arg-type]
    )
    if add_running_time_features and os.path.exists(features_running_time_path):
        with open(features_running_time_path, "r") as f:
            ft_data: dict[str, Any] = load(f)
        features_running_time = pd.DataFrame(
            ft_data["data"],
            columns=[a[0] for a in ft_data["attributes"]],  # type: ignore[arg-type]
        )
        features_running_time = features_running_time.groupby("instance_id").mean()
        if "repetition" in features_running_time.columns:
            features_running_time = features_running_time.drop(columns=["repetition"])

    # Apply PAR penalization to training data
    if training_par_factor is not None:
        performance = apply_par(performance, budget, training_par_factor)

    # Load cross-validation data
    with open(cv_path, "r") as f:
        cv_data: dict[str, Any] = load(f)
    cv = pd.DataFrame(cv_data["data"], columns=[a[0] for a in cv_data["attributes"]])  # type: ignore[arg-type]
    cv = cv.set_index("instance_id")
    if "repetition" in cv.columns:
        cv = cv.drop(columns=["repetition"])

    # Sort indices for consistency
    features = features.sort_index()  # type: ignore[attr-defined]
    performance = performance.sort_index()  # type: ignore[attr-defined]
    cv = cv.sort_index()  # type: ignore[attr-defined]
    features_running_time = features_running_time.sort_index()  # type: ignore[attr-defined]

    # Load algorithm features if available
    algorithm_features = None
    if os.path.exists(algorithm_features_path):
        with open(algorithm_features_path, "r") as f:
            af_data: dict[str, Any] = load(f)
        algorithm_features = pd.DataFrame(
            af_data["data"],
            columns=[a[0] for a in af_data["attributes"]],  # type: ignore[arg-type]
        )

        if os.path.exists(algorithm_features_runstatus_path):
            with open(algorithm_features_runstatus_path, "r") as f:
                af_rs_data: dict[str, Any] = load(f)
            af_runstatus = pd.DataFrame(
                af_rs_data["data"],
                columns=[a[0] for a in af_rs_data["attributes"]],  # type: ignore[arg-type]
            )
            for step_name, step_info in algorithm_feature_groups.items():
                if step_name in af_runstatus.columns:
                    failed_mask = af_runstatus[step_name] != "ok"
                    failed_algos = af_runstatus.loc[failed_mask, "algorithm"].values
                    step_features = step_info.get("provides", [])
                    for feat in step_features:
                        if feat in algorithm_features.columns:
                            algorithm_features.loc[
                                algorithm_features["algorithm"].isin(failed_algos), feat
                            ] = float("nan")

        algorithm_features = algorithm_features.groupby("algorithm").mean()
        algorithm_features = algorithm_features.drop(
            columns=["repetition"], errors="ignore"
        )
        algorithm_features = algorithm_features.sort_index()

    return (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    )


@overload
def evaluate_selector(
    selector_class: type,
    scenario_path: str,
    fold: int,
    hpo_func: Callable[..., Any] | None = None,
    hpo_kwargs: dict[str, Any] | None = None,
    algorithm_pre_selector: Any | None = None,
    metric: Callable[..., float] = running_time_closed_gap,
    return_per_instance: Literal[False] = False,
) -> tuple[float, Any]: ...


@overload
def evaluate_selector(
    selector_class: type,
    scenario_path: str,
    fold: int,
    hpo_func: Callable[..., Any] | None = None,
    hpo_kwargs: dict[str, Any] | None = None,
    algorithm_pre_selector: Any | None = None,
    metric: Callable[..., float] = running_time_closed_gap,
    return_per_instance: Literal[True] = True,
) -> tuple[float, Any, dict[str, float]]: ...


def evaluate_selector(
    selector_class: type,
    scenario_path: str,
    fold: int,
    hpo_func: Callable[..., Any] | None = None,
    hpo_kwargs: dict[str, Any] | None = None,
    algorithm_pre_selector: Any | None = None,
    metric: Callable[..., float] = running_time_closed_gap,
    return_per_instance: bool = False,
) -> tuple[float, Any] | tuple[float, Any, dict[str, float]]:
    """
    Runs HPO for a selector on a given ASlib scenario and fold, returns test performance.

    Parameters
    ----------
    selector_class : type
        Selector class to evaluate.
    scenario_path : str
        Path to ASlib scenario directory.
    fold : int
        Which fold to use as test set.
    hpo_func : Callable or None, default=None
        Function for HPO, must return a fitted selector.
    hpo_kwargs : dict or None, default=None
        Additional keyword arguments for hpo_func.
    algorithm_pre_selector : Any or None, default=None
        Optional preselector object.
    metric : Callable, default=running_time_closed_gap
        Metric function to evaluate the selector.
    return_per_instance : bool, default=False
        If True, also return per-instance scores dictionary.

    Returns
    -------
    test_score : float
        The test performance metric value.
    selector : Any
        The fitted selector (returned if return_per_instance is False or True).
    per_instance_scores : dict[str, float]
        Dictionary mapping instance IDs to performance (returned if return_per_instance=True).
    """
    hpo_kwargs = hpo_kwargs or {}

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
    ) = read_aslib_scenario(scenario_path)

    # Align indices
    common_idx = features.index.intersection(cv.index)
    features = features.loc[common_idx]
    performance = performance.loc[common_idx]
    cv = cv.loc[common_idx]

    # Split train and test
    train_instance_ids = cv.index[cv["fold"] != fold].unique()
    test_instance_ids = cv.index[cv["fold"] == fold].unique()

    X_train = features.loc[train_instance_ids]
    y_train = performance.loc[train_instance_ids]
    X_test = features.loc[test_instance_ids]
    y_test = performance.loc[test_instance_ids]

    features_running_time_train = features_running_time.loc[train_instance_ids]
    features_running_time_test = features_running_time.loc[test_instance_ids]

    if hpo_func is None:
        # Create base selector with provided class and params
        base_selector = selector_class(
            budget=budget, maximize=maximize, feature_groups=feature_groups
        )

        selector = SelectorPipeline(
            selector=base_selector,
            algorithm_pre_selector=algorithm_pre_selector,
            feature_groups=feature_groups,
        )

    else:
        # Run HPO (should return a fitted selector)
        selector = hpo_func(
            selector_class=selector_class,
            X=X_train,
            y=y_train,
            features_running_time=features_running_time_train,
            algorithm_features=algorithm_features,
            maximize=maximize,
            budget=budget,
            feature_groups=feature_groups,
            algorithm_pre_selector=algorithm_pre_selector,
            **hpo_kwargs,
        )

    selector.fit(X_train, y_train, algorithm_features=algorithm_features)

    # Predict and evaluate
    if selector_class is VirtualBestSolver:
        predictions = selector.predict(X_test, performance=y_test)
    else:
        predictions = selector.predict(X_test)

    test_score = metric(predictions, y_test, budget, features_running_time_test)

    if return_per_instance:
        from asf.metrics.baselines import running_time_selector_performance

        per_instance_scores = running_time_selector_performance(
            predictions,  # type: ignore[arg-type]
            y_test,
            budget,
            features_running_time_test,
            par=10,
            return_per_instance=True,
        )
        # Ensure it's a dict and cast to float
        if isinstance(per_instance_scores, dict):
            typed_scores = {k: float(v) for k, v in per_instance_scores.items()}
            return test_score, selector, typed_scores
        return test_score, selector, {}

    return test_score, selector
