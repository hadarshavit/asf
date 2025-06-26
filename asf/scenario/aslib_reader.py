import os
import pandas as pd
from asf.metrics.baselines import running_time_closed_gap


try:
    import yaml
    from yaml import SafeLoader as Loader
    # liac-arff, not arff
    from arff import load

    ASLIB_AVAILABLE = True
except ImportError:
    ASLIB_AVAILABLE = False


def read_aslib_scenario(
    path: str, add_running_time_features: bool = True
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
    features_running_time = os.path.join(path, "feature_costs.arff")
    cv_path = os.path.join(path, "cv.arff")

    # Load description file
    with open(description_path, "r") as f:
        description: dict = yaml.load(f, Loader=Loader)

    features: list[str] = description["features_deterministic"]
    feature_groups: list[str] = description["feature_steps"]
    maximize: bool = description["maximize"][0]
    budget: float = description["algorithm_cutoff_time"]

    # Load performance data
    with open(performance_path, "r") as f:
        performance: dict = load(f)
    performance = pd.DataFrame(
        performance["data"], columns=[a[0] for a in performance["attributes"]]
    )
    performance = performance.set_index("instance_id")
    # Identify which column contains runtime information
    # e.g. SAT12-INDU = "runtime", CSP-Minizinc-Obj-2016 = "time"
    runtime_col = "runtime" if "runtime" in performance.columns else "time"
    performance = performance.pivot(columns="algorithm", values=runtime_col)

    # Load feature values
    with open(features_path, "r") as f:
        features: dict = load(f)
    features = pd.DataFrame(
        features["data"], columns=[a[0] for a in features["attributes"]]
    )
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
        features_running_time = features_running_time.set_index("instance_id")

        features = pd.concat([features, features_running_time], axis=1)

    # Load cross-validation data
    with open(cv_path, "r") as f:
        cv: dict = load(f)
    cv = pd.DataFrame(cv["data"], columns=[a[0] for a in cv["attributes"]])
    cv = cv.set_index("instance_id")

    # Sort indices for consistency
    features = features.sort_index()
    performance = performance.sort_index()
    cv = cv.sort_index()

    return (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
    )


def evaluate_selector_with_hpo(
    selector_class,
    scenario_path: str,
    fold: int,
    hpo_func,
    hpo_kwargs=None,
    algorithm_pre_selector=None,  # <-- Directly pass any preselector here
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

    Returns:
        test_score: The test performance (e.g., PAR10 or other metric)
        selector: The fitted selector
    """
    if hpo_kwargs is None:
        hpo_kwargs = {}

    # Load scenario
    features, performance, features_running_time, cv, feature_groups, maximize, budget = read_aslib_scenario(scenario_path)

    # Split train/test
    X_train = features[cv["fold"] != fold]
    y_train = performance[cv["fold"] != fold]
    X_test = features[cv["fold"] == fold]
    y_test = performance[cv["fold"] == fold]

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

    # Predict and evaluate
    predictions = selector.predict(X_test)
    par = 10 #TODO: Make this configurable?
    test_score = running_time_closed_gap(predictions, y_test, budget, par, features_running_time)

    return test_score, selector


from asf.pre_selector.knee_of_the_curve_pre_selector import KneeOfCurvePreSelector
from asf.pre_selector.marginal_contribution_based import MarginalContributionBasedPreSelector
from asf.metrics.baselines import virtual_best_solver
from asf.selectors import PairwiseClassifier, MultiClassClassifier, PerformanceModel, tune_selector
from sklearn.ensemble import RandomForestClassifier
from functools import partial

if __name__ == "__main__":
    knee_preselector = KneeOfCurvePreSelector(
        metric=virtual_best_solver,
        base_pre_selector=MarginalContributionBasedPreSelector,
        maximize=True,
        S=1.0,
        workers=1,
    )

    selectors = [
        PairwiseClassifier(model_class=RandomForestClassifier),
        MultiClassClassifier(model_class=RandomForestClassifier),
        PerformanceModel(model_class=RandomForestClassifier),
    ]

    for selector in selectors:
        print(f"Evaluating selector: {selector.__class__.__name__}")

        score, selector = evaluate_selector_with_hpo(
            selector_class=partial(PairwiseClassifier, model_class=RandomForestClassifier),
            scenario_path="/home/schiller/asf/aslib_data/IPC2018",
            # scenario_path="/home/schiller/asf/aslib_data/SAT11-INDU",
            fold=1,
            hpo_func=tune_selector,
            hpo_kwargs={"runcount_limit": 10, "cv": 3, "smac_kwargs": {"overwrite": True}},
            algorithm_pre_selector=knee_preselector,
        )

        print(f"Test score: {score}")
        print(f"Selector: {selector}")
    