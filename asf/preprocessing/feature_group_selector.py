"""
Feature group selector for ASlib scenarios.

This module provides a sklearn-compatible transformer that selects features
based on feature groups defined in ASlib scenarios.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from ConfigSpace import Configuration
except ImportError:
    Configuration: Any = Any


class MissingPrerequisiteGroupError(ValueError):
    """Raised when a feature group is selected without its required prerequisite groups."""

    pass


class FeatureGroupSelector(BaseEstimator, TransformerMixin):
    """
    A sklearn-compatible transformer that selects features based on feature groups.

    This transformer filters input features to only include those belonging to
    the specified feature groups. It is designed to work with ASlib scenarios
    where features are organized into groups (feature steps).

    Parameters
    ----------
    feature_groups : dict[str, Any]
        Dictionary mapping feature group names to their metadata.
        Each value should be a dict with a 'provides' key listing the feature names
        in that group, and optionally a 'requires' key listing prerequisite groups.
    selected_groups : list[str] | None, default=None
        List of feature group names to include. If None, all groups are included.
    validate_requirements : bool, default=True
        If True, validate that all required prerequisite groups are included
        when selecting a group.

    Attributes
    ----------
    selected_features_ : list[str]
        List of feature names that will be selected after fitting.

    Examples
    --------
    >>> feature_groups = {
    ...     'basic': {'provides': ['f1', 'f2']},
    ...     'advanced': {'provides': ['f3', 'f4']}
    ... }
    >>> selector = FeatureGroupSelector(feature_groups, selected_groups=['basic'])
    >>> X = pd.DataFrame({'f1': [1], 'f2': [2], 'f3': [3], 'f4': [4]})
    >>> selector.fit_transform(X)
       f1  f2
    0   1   2

    >>> # Example with prerequisites
    >>> feature_groups = {
    ...     'Pre': {'provides': ['f1', 'f2']},
    ...     'Basic': {'provides': ['f3', 'f4'], 'requires': ['Pre']}
    ... }
    >>> selector = FeatureGroupSelector(feature_groups, selected_groups=['Basic'])
    >>> # This will raise MissingPrerequisiteGroupError because 'Pre' is not selected
    """

    def __init__(
        self,
        feature_groups: dict[str, Any],
        selected_groups: list[str] | None = None,
        validate_requirements: bool = True,
    ):
        self.feature_groups = feature_groups
        self.selected_groups = selected_groups
        self.validate_requirements = validate_requirements

        if validate_requirements and selected_groups is not None:
            self._validate_prerequisites(selected_groups)

    def _validate_prerequisites(self, selected_groups: list[str]) -> None:
        """
        Validate that all required prerequisite groups are included.

        Parameters
        ----------
        selected_groups : list[str]
            List of selected feature group names.

        Raises
        ------
        MissingPrerequisiteGroupError
            If a selected group requires another group that is not selected.
        """
        selected_set = set(selected_groups)

        for group_name in selected_groups:
            if group_name not in self.feature_groups:
                continue

            group_info = self.feature_groups[group_name]
            required_groups = group_info.get("requires", [])

            for required_group in required_groups:
                if required_group not in selected_set:
                    raise MissingPrerequisiteGroupError(
                        f"Feature group '{group_name}' requires group '{required_group}' "
                        f"to be selected, but it is not included in the selected groups. "
                        f"Selected groups: {selected_groups}"
                    )

    @staticmethod
    def validate_feature_group_selection(
        feature_groups: dict[str, Any],
        selected_groups: list[str],
    ) -> None:
        """
        Validate that a list of selected groups satisfies all prerequisites.

        This is a static utility method that can be used to validate selections
        without creating a FeatureGroupSelector instance.

        Parameters
        ----------
        feature_groups : dict[str, Any]
            Dictionary of all feature groups with their metadata.
        selected_groups : list[str]
            List of selected feature group names.

        Raises
        ------
        MissingPrerequisiteGroupError
            If a selected group requires another group that is not selected.
        """
        selected_set = set(selected_groups)

        for group_name in selected_groups:
            if group_name not in feature_groups:
                continue

            group_info = feature_groups[group_name]
            required_groups = group_info.get("requires", [])

            for required_group in required_groups:
                if required_group not in selected_set:
                    raise MissingPrerequisiteGroupError(
                        f"Feature group '{group_name}' requires group '{required_group}' "
                        f"to be selected, but it is not included in the selected groups. "
                        f"Selected groups: {list(selected_groups)}"
                    )

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureGroupSelector:
        """
        Fit the selector by determining which features to select.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : Any, default=None
            Not used, present for API compatibility.

        Returns
        -------
        FeatureGroupSelector
            The fitted selector instance.
        """
        # Determine which groups to include
        if self.selected_groups is None:
            groups_to_use = list(self.feature_groups.keys())
        else:
            groups_to_use = self.selected_groups

        # Collect features from selected groups
        selected_features = []
        for fg_name in groups_to_use:
            if fg_name in self.feature_groups:
                fg_info = self.feature_groups[fg_name]
                if "provides" in fg_info:
                    selected_features.extend(fg_info["provides"])

        # Filter to only features that exist in X
        self.selected_features_ = [f for f in selected_features if f in X.columns]

        # If no features selected, use all columns
        if not self.selected_features_:
            self.selected_features_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input by selecting only the specified features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected features.
        """
        # Filter to features that exist in X
        available_features = [f for f in self.selected_features_ if f in X.columns]
        if not available_features:
            return X
        return X[available_features]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """
        Get output feature names.

        Parameters
        ----------
        input_features : Any, default=None
            Not used, present for API compatibility.

        Returns
        -------
        list[str]
            List of selected feature names.
        """
        return self.selected_features_

    @staticmethod
    def get_selected_groups_from_config(
        feature_groups: dict[str, Any],
        config: dict[str, Any] | Configuration,
        prefix: str = "feature_group_",
    ) -> dict[str, Any] | None:
        """
        Extract selected feature groups from a SMAC configuration.

        Parameters
        ----------
        feature_groups : dict[str, Any]
            Dictionary of all feature groups.
        config : dict[str, Any]
            SMAC configuration dictionary.
        prefix : str, default="feature_group_"
            Prefix used for feature group parameters in the config.

        Returns
        -------
        dict[str, Any] or None
            Dictionary of selected feature groups, or None if no groups selected.
        """
        selected = {}
        for fg_name, fg_info in feature_groups.items():
            if config.get(f"{prefix}{fg_name}", True):
                selected[fg_name] = fg_info
        return selected if selected else None
