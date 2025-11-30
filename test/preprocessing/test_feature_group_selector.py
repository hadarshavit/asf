"""Tests for FeatureGroupSelector and prerequisite validation."""

import pytest
import pandas as pd

from asf.preprocessing.feature_group_selector import (
    FeatureGroupSelector,
    MissingPrerequisiteGroupError,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def feature_groups_with_prereqs():
    """Feature groups with prerequisite dependencies (like SAT12-INDU)."""
    return {
        "Pre": {"provides": ["nvars", "nclauses", "reducedVars"]},
        "Basic": {
            "provides": ["vars_clauses_ratio", "POSNEG_RATIO"],
            "requires": ["Pre"],
        },
        "KLB": {"provides": ["VCG_VAR_mean", "VG_mean"], "requires": ["Pre"]},
        "CG": {"provides": ["CG_mean", "cluster_coeff"], "requires": ["Pre"]},
        "Advanced": {"provides": ["adv_feature"], "requires": ["Pre", "Basic"]},
    }


@pytest.fixture
def feature_groups_no_prereqs():
    """Feature groups without prerequisites."""
    return {
        "basic": {"provides": ["f1", "f2"]},
        "advanced": {"provides": ["f3", "f4"]},
    }


@pytest.fixture
def sample_features_df():
    """Sample feature DataFrame."""
    return pd.DataFrame(
        {
            "nvars": [100, 200],
            "nclauses": [300, 400],
            "reducedVars": [10, 20],
            "vars_clauses_ratio": [0.33, 0.5],
            "POSNEG_RATIO": [0.5, 0.6],
            "VCG_VAR_mean": [0.1, 0.2],
            "VG_mean": [0.3, 0.4],
            "CG_mean": [0.5, 0.6],
            "cluster_coeff": [0.7, 0.8],
            "adv_feature": [0.9, 1.0],
        },
        index=["inst1", "inst2"],
    )


# ============================================================================
# FeatureGroupSelector Prerequisite Validation Tests
# ============================================================================


class TestFeatureGroupSelectorPrerequisites:
    """Tests for prerequisite validation in FeatureGroupSelector."""

    def test_valid_selection_with_prereqs(self, feature_groups_with_prereqs):
        """Selecting groups with all prerequisites should work."""
        # Pre has no prereqs
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre"]
        )
        assert selector.selected_groups == ["Pre"]

        # Basic requires Pre
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre", "Basic"]
        )
        assert selector.selected_groups == ["Pre", "Basic"]

        # Multiple groups requiring Pre
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre", "Basic", "KLB", "CG"]
        )
        assert "Pre" in selector.selected_groups

    def test_invalid_selection_missing_prereq(self, feature_groups_with_prereqs):
        """Selecting a group without its prerequisite should raise error."""
        with pytest.raises(MissingPrerequisiteGroupError) as exc_info:
            FeatureGroupSelector(feature_groups_with_prereqs, selected_groups=["Basic"])

        assert "Basic" in str(exc_info.value)
        assert "Pre" in str(exc_info.value)

    def test_invalid_selection_missing_chained_prereq(
        self, feature_groups_with_prereqs
    ):
        """Selecting Advanced without Basic should raise error (Advanced requires Pre and Basic)."""
        with pytest.raises(MissingPrerequisiteGroupError) as exc_info:
            FeatureGroupSelector(
                feature_groups_with_prereqs, selected_groups=["Pre", "Advanced"]
            )

        assert "Advanced" in str(exc_info.value)
        assert "Basic" in str(exc_info.value)

    def test_valid_full_chain(self, feature_groups_with_prereqs):
        """Selecting Advanced with all prerequisites should work."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre", "Basic", "Advanced"]
        )
        assert "Advanced" in selector.selected_groups

    def test_validation_disabled(self, feature_groups_with_prereqs):
        """When validation is disabled, invalid selections should be allowed."""
        # This should NOT raise even though Basic requires Pre
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs,
            selected_groups=["Basic"],
            validate_requirements=False,
        )
        assert selector.selected_groups == ["Basic"]

    def test_none_selected_groups_no_validation(self, feature_groups_with_prereqs):
        """When selected_groups is None, no validation should occur."""
        # Should not raise - None means all groups
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=None
        )
        assert selector.selected_groups is None

    def test_no_prereqs_any_selection_valid(self, feature_groups_no_prereqs):
        """Groups without prerequisites can be selected in any combination."""
        selector = FeatureGroupSelector(
            feature_groups_no_prereqs, selected_groups=["advanced"]
        )
        assert selector.selected_groups == ["advanced"]

        selector = FeatureGroupSelector(
            feature_groups_no_prereqs, selected_groups=["basic", "advanced"]
        )
        assert set(selector.selected_groups) == {"basic", "advanced"}


# ============================================================================
# FeatureGroupSelector Transform Tests
# ============================================================================


class TestFeatureGroupSelectorTransform:
    """Tests for feature selection transform functionality."""

    def test_fit_transform_selects_correct_features(
        self, feature_groups_with_prereqs, sample_features_df
    ):
        """fit_transform should select only features from selected groups."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre"]
        )
        result = selector.fit_transform(sample_features_df)

        assert list(result.columns) == ["nvars", "nclauses", "reducedVars"]
        assert len(result) == 2

    def test_fit_transform_multiple_groups(
        self, feature_groups_with_prereqs, sample_features_df
    ):
        """fit_transform with multiple groups should include all their features."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre", "Basic"]
        )
        result = selector.fit_transform(sample_features_df)

        expected_features = [
            "nvars",
            "nclauses",
            "reducedVars",
            "vars_clauses_ratio",
            "POSNEG_RATIO",
        ]
        assert list(result.columns) == expected_features

    def test_fit_transform_none_selects_all(
        self, feature_groups_with_prereqs, sample_features_df
    ):
        """When selected_groups is None, all group features should be selected."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=None
        )
        result = selector.fit_transform(sample_features_df)

        # Should have all features from all groups
        all_provided = []
        for fg_info in feature_groups_with_prereqs.values():
            all_provided.extend(fg_info["provides"])

        # Only features that exist in the DataFrame
        expected = [f for f in all_provided if f in sample_features_df.columns]
        assert list(result.columns) == expected

    def test_transform_handles_missing_features(self, feature_groups_with_prereqs):
        """Transform should handle DataFrames missing some expected features."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre"]
        )

        # DataFrame with only some of the expected features
        X_train = pd.DataFrame({"nvars": [100], "nclauses": [200], "reducedVars": [10]})
        X_test = pd.DataFrame(
            {"nvars": [150], "nclauses": [250]}
        )  # Missing reducedVars

        selector.fit(X_train)
        result = selector.transform(X_test)

        # Should only include features present in X_test
        assert list(result.columns) == ["nvars", "nclauses"]

    def test_get_feature_names_out(
        self, feature_groups_with_prereqs, sample_features_df
    ):
        """get_feature_names_out should return selected feature names."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs, selected_groups=["Pre", "Basic"]
        )
        selector.fit(sample_features_df)

        feature_names = selector.get_feature_names_out()
        expected = [
            "nvars",
            "nclauses",
            "reducedVars",
            "vars_clauses_ratio",
            "POSNEG_RATIO",
        ]
        assert feature_names == expected


# ============================================================================
# Static Validation Method Tests
# ============================================================================


class TestStaticValidation:
    """Tests for the static validation method."""

    def test_validate_feature_group_selection_valid(self, feature_groups_with_prereqs):
        """Static validation should pass for valid selections."""
        # Should not raise
        FeatureGroupSelector.validate_feature_group_selection(
            feature_groups_with_prereqs, ["Pre", "Basic", "KLB"]
        )

    def test_validate_feature_group_selection_invalid(
        self, feature_groups_with_prereqs
    ):
        """Static validation should raise for invalid selections."""
        with pytest.raises(MissingPrerequisiteGroupError):
            FeatureGroupSelector.validate_feature_group_selection(
                feature_groups_with_prereqs,
                ["Basic", "KLB"],  # Missing Pre
            )

    def test_get_selected_groups_from_config(self, feature_groups_with_prereqs):
        """get_selected_groups_from_config should extract enabled groups."""
        config = {
            "feature_group_Pre": True,
            "feature_group_Basic": True,
            "feature_group_KLB": False,
            "feature_group_CG": True,
            "feature_group_Advanced": False,
        }

        selected = FeatureGroupSelector.get_selected_groups_from_config(
            feature_groups_with_prereqs, config
        )

        assert set(selected) == {"Pre", "Basic", "CG"}

    def test_get_selected_groups_from_config_all_false(
        self, feature_groups_with_prereqs
    ):
        """When all groups are disabled, should return None."""
        config = {
            "feature_group_Pre": False,
            "feature_group_Basic": False,
            "feature_group_KLB": False,
            "feature_group_CG": False,
            "feature_group_Advanced": False,
        }

        selected = FeatureGroupSelector.get_selected_groups_from_config(
            feature_groups_with_prereqs, config
        )

        assert selected is None

    def test_get_selected_groups_from_config_default_true(
        self, feature_groups_with_prereqs
    ):
        """Missing config entries should default to True."""
        config = {
            "feature_group_Pre": False,
            # Other entries missing - should default to True
        }

        selected = FeatureGroupSelector.get_selected_groups_from_config(
            feature_groups_with_prereqs, config
        )

        # Pre is False, others default to True
        assert "Pre" not in selected
        assert "Basic" in selected


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_feature_groups(self):
        """Empty feature_groups dict should work."""
        selector = FeatureGroupSelector({}, selected_groups=None)
        X = pd.DataFrame({"f1": [1, 2]})
        result = selector.fit_transform(X)
        assert list(result.columns) == ["f1"]  # Falls back to all columns

    def test_unknown_group_in_selection(self, feature_groups_with_prereqs):
        """Unknown groups in selection should be ignored gracefully."""
        selector = FeatureGroupSelector(
            feature_groups_with_prereqs,
            selected_groups=["Pre", "UnknownGroup"],
            validate_requirements=True,
        )
        X = pd.DataFrame({"nvars": [100], "nclauses": [200], "reducedVars": [10]})
        result = selector.fit_transform(X)

        # Should only include features from Pre (UnknownGroup is ignored)
        assert list(result.columns) == ["nvars", "nclauses", "reducedVars"]

    def test_group_with_empty_provides(self, feature_groups_with_prereqs):
        """Groups with empty provides list should be handled."""
        feature_groups = {
            "Pre": {"provides": []},
            "Basic": {"provides": ["f1"], "requires": ["Pre"]},
        }
        selector = FeatureGroupSelector(
            feature_groups, selected_groups=["Pre", "Basic"]
        )
        X = pd.DataFrame({"f1": [1]})
        result = selector.fit_transform(X)
        assert list(result.columns) == ["f1"]

    def test_circular_dependency_detection(self):
        """Currently circular dependencies aren't explicitly checked, but shouldn't cause infinite loop."""
        feature_groups = {
            "A": {"provides": ["f1"], "requires": ["B"]},
            "B": {"provides": ["f2"], "requires": ["A"]},
        }
        # This would require both A and B, creating a circular dependency
        # The validation should fail since neither can be selected alone
        with pytest.raises(MissingPrerequisiteGroupError):
            FeatureGroupSelector(feature_groups, selected_groups=["A"])

        with pytest.raises(MissingPrerequisiteGroupError):
            FeatureGroupSelector(feature_groups, selected_groups=["B"])

        # But selecting both should work
        selector = FeatureGroupSelector(feature_groups, selected_groups=["A", "B"])
        assert set(selector.selected_groups) == {"A", "B"}
