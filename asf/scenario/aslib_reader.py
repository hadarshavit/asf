import os
import pandas as pd
from asf.metrics.baselines import running_time_closed_gap
from asf.metrics.par10 import apply_par
from asf.selectors.selector_pipeline import SelectorPipeline


try:
    import yaml
    from yaml import SafeLoader as Loader

    from arff import load

    ASLIB_AVAILABLE = True
except ImportError:
    ASLIB_AVAILABLE = False


def read_aslib_scenario(
    path: str,
    add_running_time_features: bool = True,
    training_par_factor: float | None = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], bool, float]:
    """Read an ASlib scenario from a file.

    Args:
        path (str): The path to the ASlib scenario directory.
        add_running_time_features (bool, optional): Whether to include running time features. Defaults to True.

    Returns:
        # TODO: Update the return type annotation to match the actual return type.
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], bool, float]:
            - features (pd.DataFrame): A DataFrame containing the feature values for each instance.
            - performance (pd.DataFrame): A DataFrame containing the performance data for each algorithm and instance.
            - cv (pd.DataFrame): A DataFrame containing cross-validation data.
            - feature_groups (list[str]): A list of feature groups defined in the scenario.
            - maximize (bool): A flag indicating whether the objective is to maximize performance.
            - budget (float): The algorithm cutoff time or budget for the scenario.

    Raises:
        ImportError: If the required ASlib library is not available.
    """
    if not ASLIB_AVAILABLE:
        raise ImportError(
            "The aslib library is not available. Install it via 'pip install asf-lib[aslib]'."
        )

    description_path = os.path.join(path, "description.txt")
    performance_path = os.path.join(path, "algorithm_runs.arff")
    features_path = os.path.join(path, "feature_values.arff")
    features_runstatus_path = os.path.join(path, "feature_runstatus.arff")
    features_running_time = os.path.join(path, "feature_costs.arff")
    cv_path = os.path.join(path, "cv.arff")
    algorithm_features_path = os.path.join(path, "algorithm_feature_values.arff")
    algorithm_features_runstatus_path = os.path.join(
        path, "algorithm_feature_runstatus.arff"
    )

    # Load description file
    with open(description_path, "r") as f:
        description: dict = yaml.load(f, Loader=Loader)

    features: list[str] = description["features_deterministic"]
    feature_groups: list[str] = description["feature_steps"]
    algorithm_feature_groups: dict = description.get("algorithm_feature_steps", {})
    maximize: bool = description["maximize"][0]
    budget: float = description["algorithm_cutoff_time"]

    # Load performance data
    with open(performance_path, "r") as f:
        performance: dict = load(f)
    performance = pd.DataFrame(
        performance["data"], columns=[a[0] for a in performance["attributes"]]
    )

    # Identify which column contains runtime information
    # e.g. SAT12-INDU = "runtime", CSP-Minizinc-Obj-2016 = "time"
    runtime_col = description["performance_measures"][0]

    # Handle failed runs: if runstatus is not "ok", set runtime to budget (timeout)
    # This ensures crashes, memouts, and other failures are treated as timeouts
    if "runstatus" in performance.columns:
        performance.loc[performance["runstatus"] != "ok", runtime_col] = budget

    # Aggregate over repetitions (e.g., take mean)
    group_cols = ["instance_id", "algorithm"]
    performance = performance.groupby(group_cols, as_index=False)[runtime_col].mean()

    performance = performance.pivot(
        index="instance_id", columns="algorithm", values=runtime_col
    )

    # Load feature values
    with open(features_path, "r") as f:
        features: dict = load(f)
    features = pd.DataFrame(
        features["data"], columns=[a[0] for a in features["attributes"]]
    )

    # Load feature runstatus and set features to NaN where runstatus != "ok"
    if os.path.exists(features_runstatus_path):
        with open(features_runstatus_path, "r") as f:
            feature_runstatus: dict = load(f)
        feature_runstatus = pd.DataFrame(
            feature_runstatus["data"],
            columns=[a[0] for a in feature_runstatus["attributes"]],
        )
        # For each feature step, if runstatus != "ok", set corresponding features to NaN
        # Get feature step to features mapping from description
        for step_name, step_info in feature_groups.items():
            if step_name in feature_runstatus.columns:
                # Find instances where this step failed
                failed_mask = feature_runstatus[step_name] != "ok"
                failed_instances = feature_runstatus.loc[
                    failed_mask, "instance_id"
                ].values
                # Get features belonging to this step
                step_features = step_info.get("provides", [])
                # Set those features to NaN for failed instances
                for feat in step_features:
                    if feat in features.columns:
                        features.loc[
                            features["instance_id"].isin(failed_instances), feat
                        ] = float("nan")

    features = features.groupby("instance_id").mean()
    features = features.drop(columns=["repetition"])

    # Optionally load running time features
    if add_running_time_features:
        with open(features_running_time, "r") as f:
            features_running_time: dict = load(f)
        features_running_time = pd.DataFrame(
            features_running_time["data"],
            columns=[a[0] for a in features_running_time["attributes"]],
        )
        features_running_time = features_running_time.groupby("instance_id").mean()

        features.index.name = "instance_id"
        features_running_time.index.name = "instance_id"

        features = pd.concat([features, features_running_time], axis=1)

    # Apply PAR penalization to training data so the selector learns to avoid timeouts
    if training_par_factor is not None:
        performance = apply_par(performance, budget, training_par_factor)

    # Load cross-validation data
    with open(cv_path, "r") as f:
        cv: dict = load(f)
    cv = pd.DataFrame(cv["data"], columns=[a[0] for a in cv["attributes"]])
    cv = cv.set_index("instance_id")

    # Sort indices for consistency
    features = features.sort_index()
    performance = performance.sort_index()
    cv = cv.sort_index()

    # Load algorithm features if available
    algorithm_features = None
    if os.path.exists(algorithm_features_path):
        with open(algorithm_features_path, "r") as f:
            algorithm_features_data: dict = load(f)
        algorithm_features = pd.DataFrame(
            algorithm_features_data["data"],
            columns=[a[0] for a in algorithm_features_data["attributes"]],
        )

        # Load algorithm feature runstatus and set features to NaN where runstatus != "ok"
        if os.path.exists(algorithm_features_runstatus_path):
            with open(algorithm_features_runstatus_path, "r") as f:
                algorithm_feature_runstatus: dict = load(f)
            algorithm_feature_runstatus = pd.DataFrame(
                algorithm_feature_runstatus["data"],
                columns=[a[0] for a in algorithm_feature_runstatus["attributes"]],
            )
            # For each feature step, if runstatus != "ok", set corresponding features to NaN
            for step_name, step_info in algorithm_feature_groups.items():
                if step_name in algorithm_feature_runstatus.columns:
                    # Find algorithms where this step failed
                    failed_mask = algorithm_feature_runstatus[step_name] != "ok"
                    failed_algorithms = algorithm_feature_runstatus.loc[
                        failed_mask, "algorithm"
                    ].values
                    # Get features belonging to this step
                    step_features = step_info.get("provides", [])
                    # Set those features to NaN for failed algorithms
                    for feat in step_features:
                        if feat in algorithm_features.columns:
                            algorithm_features.loc[
                                algorithm_features["algorithm"].isin(failed_algorithms),
                                feat,
                            ] = float("nan")

        # Aggregate over repetitions (e.g., take mean) and set algorithm as index
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


def evaluate_selector(
    selector_class,
    scenario_path: str,
    fold: int,
    hpo_func=None,
    hpo_kwargs={},
    algorithm_pre_selector=None,
    metric=running_time_closed_gap,
):
    """
    Runs HPO for a selector on a given ASlib scenario and fold, returns test performance.

    Args:
        selector_class: Selector class or callable
        scenario_path: Path to ASlib scenario
        fold: Which fold to use as test
        hpo_func: Function for HPO, must return a fitted selector
        hpo_kwargs: Optional dict of extra kwargs for HPO
        algorithm_pre_selector: Optional preselector object (e.g., KneeOfCurvePreSelector instance)
        metric: Metric function to evaluate the selector
        training_par_factor: PAR factor to apply to training performance data. Timeouts
            (values > budget) are replaced with budget * training_par_factor. This helps
            the selector learn to avoid timeouts. Set to None to disable. Defaults to 10.0.

    Returns:
        test_score: The test performance (e.g., PAR10 or other metric)
        selector: The fitted selector
    """

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

    if hpo_func is None:
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
            maximize=maximize,
            budget=budget,
            feature_groups=feature_groups,
            algorithm_pre_selector=algorithm_pre_selector,
            **hpo_kwargs,
        )

    selector.fit(X_train, y_train, algorithm_features=algorithm_features)

    # Predict and evaluate
    predictions = selector.predict(X_test)
    test_score = metric(predictions, y_test, budget, features_running_time)

    return test_score, selector
