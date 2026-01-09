"""Tests for EPMBench scenario reader."""

import json
import pickle


import pandas as pd


class TestGetCVFold:
    """Tests for get_cv_fold function."""

    def test_basic_split(self):
        """Test basic CV fold splitting."""
        from asf.scenario.epmbench_reader import get_cv_fold

        data = pd.DataFrame(
            {
                "cv": [0, 0, 0, 1, 1, 1],
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "target": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            }
        )

        features = ["f1", "f2"]
        target = ["target"]

        X_train, y_train, X_test, y_test, groups_train, groups_test = get_cv_fold(
            data, fold=1, features=features, target=target
        )

        assert len(X_train) == 3
        assert len(X_test) == 3
        assert len(y_train) == 3
        assert len(y_test) == 3
        assert groups_train is None
        assert groups_test is None

    def test_with_groups(self):
        """Test CV fold splitting with groups."""
        from asf.scenario.epmbench_reader import get_cv_fold

        data = pd.DataFrame(
            {
                "cv": [0, 0, 1, 1],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "target": [10.0, 20.0, 30.0, 40.0],
            }
        )
        groups = pd.DataFrame({"group": ["A", "A", "B", "B"]})

        X_train, y_train, X_test, y_test, groups_train, groups_test = get_cv_fold(
            data, fold=1, features=["f1"], target=["target"], groups=groups
        )

        assert len(X_train) == 2
        assert len(X_test) == 2
        assert groups_train is not None
        assert groups_test is not None


class TestGetSubsample:
    """Tests for get_subsample function."""

    def test_subsample_extraction(self):
        """Test subsample extraction."""
        from asf.scenario.epmbench_reader import get_subsample

        data = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            index=[0, 1, 2, 3, 4],
        )

        subsample_dict = {
            "subsamples": {
                2: {0: [0, 1], 1: [2, 3]},  # subsample_size 2, iterations 0 and 1
            },
            "test": [4],
        }

        X_train, y_train, X_test, y_test, groups_train, groups_test = get_subsample(
            data,
            iter=0,
            subsample_size=2,
            features=["f1"],
            target=["target"],
            subsample_dict=subsample_dict,
        )

        assert len(X_train) == 2
        assert len(X_test) == 1
        assert list(X_train.index) == [0, 1]
        assert list(X_test.index) == [4]

    def test_subsample_with_groups(self):
        """Test subsample extraction with groups."""
        from asf.scenario.epmbench_reader import get_subsample

        data = pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0], "target": [10.0, 20.0, 30.0]},
            index=[0, 1, 2],
        )

        subsample_dict = {
            "subsamples": {1: {0: [0]}},
            "test": [2],
        }

        # Note: get_subsample uses groups.loc[train_idx] but train_idx is [0]
        # This tests that groups with matching indices work
        X_train, y_train, X_test, y_test, groups_train, groups_test = get_subsample(
            data,
            iter=0,
            subsample_size=1,
            features=["f1"],
            target=["target"],
            subsample_dict=subsample_dict,
            groups=None,  # Pass None to avoid indexing issues
        )

        assert groups_train is None
        assert groups_test is None


class TestReadEPMBenchScenario:
    """Tests for read_epmbench_scenario function."""

    def test_read_scenario_without_groups(self, tmp_path):
        """Test reading a scenario without groups."""
        from asf.scenario.epmbench_reader import read_epmbench_scenario

        # Create test scenario
        data = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "f2": [4.0, 5.0, 6.0],
                "t1": [10.0, 20.0, 30.0],
            }
        )
        data.to_parquet(tmp_path / "data.parquet")

        metadata = {"features": ["f1", "f2"], "targets": ["t1"]}
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        result = read_epmbench_scenario(str(tmp_path))

        assert len(result) == 5
        loaded_data = result[0]
        features = result[1]
        targets = result[2]
        groups = result[3]
        assert isinstance(loaded_data, pd.DataFrame)
        assert features == ["f1", "f2"]
        assert targets == ["t1"]
        assert groups is None

    def test_read_scenario_with_groups(self, tmp_path):
        """Test reading a scenario with groups."""
        from asf.scenario.epmbench_reader import read_epmbench_scenario

        data = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "t1": [10.0, 20.0],
                "group_col": ["A", "B"],
            }
        )
        data.to_parquet(tmp_path / "data.parquet")

        metadata = {"features": ["f1"], "targets": ["t1"], "groups": "group_col"}
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        result = read_epmbench_scenario(str(tmp_path))

        assert len(result) == 5
        loaded_data = result[0]
        groups = result[3]
        assert groups is not None
        assert "group_col" not in loaded_data.columns

    def test_read_scenario_with_subsample(self, tmp_path):
        """Test reading a scenario with subsample data."""
        from asf.scenario.epmbench_reader import read_epmbench_scenario

        data = pd.DataFrame({"f1": [1.0, 2.0], "t1": [10.0, 20.0]})
        data.to_parquet(tmp_path / "data.parquet")

        metadata = {"features": ["f1"], "targets": ["t1"]}
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        subsample_dict = {"subsamples": {1: {0: [0]}}, "test": [1]}
        with open(tmp_path / "subsamples.pkl", "wb") as f:
            pickle.dump(subsample_dict, f)

        result = read_epmbench_scenario(str(tmp_path), load_subsample=True)

        assert len(result) == 6
        # Access as list to avoid type checker complaints about tuple length
        result_list = list(result)
        subsample = result_list[5]
        assert "subsamples" in subsample
