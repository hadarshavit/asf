"""Additional tests for selector pipeline to improve coverage."""

import pandas as pd

from asf.selectors.selector_pipeline import SelectorPipeline
from asf.selectors.baselines import SingleBestSolver


class TestSelectorPipelineExtended:
    """Extended tests for SelectorPipeline."""

    def test_basic_pipeline(self):
        """Test basic pipeline with just a selector."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 2.0, 3.0], "algo2": [3.0, 2.0, 1.0]},
            index=["i1", "i2", "i3"],
        )

        pipeline = SelectorPipeline(selector=SingleBestSolver())
        pipeline.fit(features=features, performance=performance)
        predictions = pipeline.predict(features=features)

        assert isinstance(predictions, dict)
        assert len(predictions) == 3


class TestSelectorPipelineSaveLoad:
    """Tests for pipeline save/load functionality."""

    def test_save(self, tmp_path):
        """Test saving a pipeline."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 2.0, 3.0], "algo2": [3.0, 2.0, 1.0]},
            index=["i1", "i2", "i3"],
        )

        pipeline = SelectorPipeline(selector=SingleBestSolver())
        pipeline.fit(features=features, performance=performance)

        save_path = str(tmp_path / "pipeline.pkl")
        pipeline.save(save_path)

        import os

        assert os.path.exists(save_path)

    def test_load(self, tmp_path):
        """Test loading a pipeline."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 2.0, 3.0], "algo2": [3.0, 2.0, 1.0]},
            index=["i1", "i2", "i3"],
        )

        pipeline = SelectorPipeline(selector=SingleBestSolver())
        pipeline.fit(features=features, performance=performance)

        save_path = str(tmp_path / "pipeline.pkl")
        pipeline.save(save_path)

        loaded = SelectorPipeline.load(save_path)
        loaded_predictions = loaded.predict(features=features)

        assert isinstance(loaded_predictions, dict)
