import json
import os
import pickle
from typing import Any

import pandas as pd


def read_epmbench_scenario(
    path: str, load_subsample: bool = False
) -> (
    tuple[pd.DataFrame, list[str], list[str], pd.DataFrame | None, dict[str, Any]]
    | tuple[
        pd.DataFrame,
        list[str],
        list[str],
        pd.DataFrame | None,
        dict[str, Any],
        dict[str, Any],
    ]
):
    """
    Reads the EPMBench scenario from the given path.

    Args:
        path (str): Path to the EPMBench scenario directory.
        load_subsample (bool, optional): Whether to load subsample data. Defaults to False.

    Returns:
    tuple[pd.DataFrame, list[str], list[str], pd.DataFrame | None, dict[str, Any]]
    | tuple[pd.DataFrame, list[str], list[str], pd.DataFrame | None, dict[str, Any], dict[str, Any]]:
              If `load_subsample` is False, returns a tuple containing:
                - data (pd.DataFrame): The main dataset.
        - features (list[str]): List of feature names.
        - targets (list[str]): List of target names.
        - groups (pd.DataFrame | None): Group information if available, otherwise None.
        - metadata (dict[str, Any]): Metadata dictionary.
              If `load_subsample` is True, an additional subsample dictionary is included in the tuple.
    """
    with open(os.path.join(path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    data = pd.read_parquet(os.path.join(path, "data.parquet"))
    if "groups" in metadata:
        groups = data[metadata["groups"]]
        data.drop(columns=[metadata["groups"]], inplace=True)
    else:
        groups = None

    if load_subsample:
        with open(os.path.join(path, "subsamples.pkl"), "rb") as f:
            subsample_dict = pickle.load(f)

    if not load_subsample:
        return data, metadata["features"], metadata["targets"], groups, metadata
    else:
        return (
            data,
            metadata["features"],
            metadata["targets"],
            groups,
            metadata,
            subsample_dict,
        )


def get_cv_fold(
    data: pd.DataFrame,
    fold: int,
    features: list[str],
    target: list[str],
    groups: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    Splits the data into training and testing sets based on the specified fold.

    Args:
        data (pd.DataFrame): The dataset.
        fold (int): The fold number.
    features (list[str]): List of feature names.
    target (list[str]): List of target names.
    groups (pd.DataFrame | None, optional): Group information if available. Defaults to None.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.DataFrame): Training targets.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.DataFrame): Testing targets.
            - groups_train (Optional[pd.DataFrame]): Training groups if available, otherwise None.
            - groups_test (Optional[pd.DataFrame]): Testing groups if available, otherwise None.
    """
    train_idx = data["cv"] != fold
    test_idx = data["cv"] == fold

    train_data = data[train_idx]
    test_data = data[test_idx]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    if groups is not None:
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
    else:
        groups_train = None
        groups_test = None

    return X_train, y_train, X_test, y_test, groups_train, groups_test


def get_subsample(
    data: pd.DataFrame,
    iter: int,
    subsample_size: int,
    features: list[str],
    target: list[str],
    subsample_dict: dict[str, Any],
    groups: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    Splits the data into training and testing sets based on the specified subsample iteration.

    Args:
        data (pd.DataFrame): The dataset.
        iter (int): The iteration number.
        subsample_size (int): The size of the subsample.
    features (list[str]): List of feature names.
    target (list[str]): List of target names.
    subsample_dict (dict[str, Any]): Dictionary containing subsample indices.
    groups (pd.DataFrame | None, optional): Group information if available. Defaults to None.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.DataFrame): Training targets.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.DataFrame): Testing targets.
            - groups_train (Optional[pd.DataFrame]): Training groups if available, otherwise None.
            - groups_test (Optional[pd.DataFrame]): Testing groups if available, otherwise None.
    """
    train_idx = subsample_dict["subsamples"][subsample_size][iter]
    test_idx = subsample_dict["test"]

    train_data = data.loc[train_idx]
    test_data = data.loc[test_idx]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    if groups is not None:
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
    else:
        groups_train = None
        groups_test = None

    return X_train, y_train, X_test, y_test, groups_train, groups_test
