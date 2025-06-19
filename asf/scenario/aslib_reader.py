import os
import pandas as pd

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
    performance = performance.pivot(columns="algorithm", values="time")

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
