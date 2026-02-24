import numpy as np
import pandas as pd
import os
from ConfigSpace import Configuration

from asf.selectors.selector_pipeline import SelectorPipeline
from asf.selectors.multi_class import MultiClassClassifier
from asf.selectors.baselines import SingleBestSolver
from asf.predictors.random_forest import RandomForestClassifierWrapper
from asf.utils.configurable import (
    ConfigurableMixin,
    convert_class_choices_to_categorical,
)


# Dummy classes for config tests
class DummySelector(ConfigurableMixin):
    PREFIX = "dummy_sel"

    def __init__(self, **kwargs):
        self.budget = 10.0

    @staticmethod
    def _define_hyperparameters(**kwargs):
        return [], [], []


class DummyPreprocessor1(ConfigurableMixin):
    PREFIX = "p1"
    pass


class DummyPreprocessor2(ConfigurableMixin):
    PREFIX = "p2"
    pass


def small_dataframes():
    X = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [0.5, 0.7, np.nan, 1.2],
        },
        index=pd.Index([f"i{k}" for k in range(4)]),
    )
    Y = pd.DataFrame(
        {
            "a": [5.0, 2.0, 1.0, 4.0],
            "b": [1.0, 3.0, 2.0, 6.0],
        },
        index=X.index,
    )
    return X, Y


class TestSelectorPipeline:
    """Consolidated tests for SelectorPipeline."""

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

    def test_selector_pipeline_fit_predict_and_config(self, tmp_path):
        """Test pipeline with multi-class classifier and preprocessing."""
        X, Y = small_dataframes()

        selector = MultiClassClassifier(
            model_class=RandomForestClassifierWrapper, budget=10.0
        )
        pipe = SelectorPipeline(selector=selector)

        # Fit with NaNs to exercise SimpleImputer in pipeline
        pipe.fit(X, Y)
        preds = pipe.predict(X)

        assert set(preds.keys()) == set(X.index)
        for v in preds.values():
            assert isinstance(v, list) and len(v) == 1
            algo, bud, *_ = v[0]
            assert algo in ["a", "b"]
            assert bud == 10.0

        # Config should reflect selector & preprocessing steps
        cfg = pipe.get_config()
        assert cfg["selector"] == "MultiClassClassifier"
        assert cfg["selector_model"] in ("RandomForestClassifierWrapper",)
        assert "SimpleImputer" in cfg["preprocessor_steps"][0]

        # Save/load round-trip
        pth = tmp_path / "pipe.joblib"
        pipe.save(pth.as_posix())
        loaded = SelectorPipeline.load(pth.as_posix())
        preds2 = loaded.predict(X)
        assert set(preds2.keys()) == set(X.index)

    def test_save_load(self, tmp_path):
        """Test pipeline save and load logic."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        performance = pd.DataFrame(
            {"algo1": [1.0, 2.0, 3.0], "algo2": [3.0, 2.0, 1.0]},
            index=["i1", "i2", "i3"],
        )

        pipeline = SelectorPipeline(selector=SingleBestSolver())
        pipeline.fit(features=features, performance=performance)

        save_path = str(tmp_path / "pipeline.pkl")
        pipeline.save(save_path)
        assert os.path.exists(save_path)

        loaded = SelectorPipeline.load(save_path)
        loaded_predictions = loaded.predict(features=features)
        assert isinstance(loaded_predictions, dict)
        assert len(loaded_predictions) == 3


class TestSelectorPipelineConfig:
    """Tests for SelectorPipeline configuration and auto-discovery."""

    def test_config_space_refactor(self):
        # Define space
        cs = SelectorPipeline.get_configuration_space(
            selector_class=[DummySelector],
            preprocessing_class=[DummyPreprocessor1, DummyPreprocessor2],
        )

        # Check HPs
        hps = dict(cs)

        # Should have preprocessor:DummyPreprocessor1
        hp_name_1 = f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1"
        assert hp_name_1 in hps
        hp1 = hps[hp_name_1]

        # Check choices
        assert hasattr(hp1, "choices")
        assert "DummyPreprocessor1" in hp1.choices  # type: ignore[attr-defined]
        assert "False" in hp1.choices  # type: ignore[attr-defined]
        assert hp1.default_value == "False"

        # Sample configuration
        config_dict = {
            hp_name_1: "DummyPreprocessor1",
            f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor2": "False",
            f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
        }
        config = Configuration(cs, values=config_dict)

        # Create pipeline
        pipeline_partial = SelectorPipeline.get_from_configuration(config)
        pipeline = pipeline_partial()

        assert pipeline.preprocessor is not None
        steps = pipeline.preprocessor.steps
        step_names = [s[0] for s in steps]

        assert "SimpleImputer" in step_names[0]
        assert "DummyPreprocessor1" in step_names
        assert "DummyPreprocessor2" not in step_names

        # Verify config retrieval
        cfg = pipeline.get_config()
        assert "DummyPreprocessor1" in cfg["preprocessor_steps"]

    def test_none_preprocessing(self):
        # Test with no preprocessors active
        cs = SelectorPipeline.get_configuration_space(
            selector_class=[DummySelector], preprocessing_class=[DummyPreprocessor1]
        )

        config_dict = {
            f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1": "False",
            f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
        }
        config = Configuration(cs, values=config_dict)

        pipeline = SelectorPipeline.get_from_configuration(config)()

        # Should only have SimpleImputer
        steps = pipeline.preprocessor.steps
        assert len(steps) == 1
        assert steps[0][0] == "SimpleImputer"

    def test_auto_discovery(self):
        # Test that we can recover classes from the space without passing them explicitly
        cs = SelectorPipeline.get_configuration_space(
            selector_class=[DummySelector],
            preprocessing_class=[DummyPreprocessor1, DummyPreprocessor2],
        )

        config_dict = {
            f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1": "DummyPreprocessor1",
            f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor2": "False",
            f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
        }
        config = Configuration(cs, values=config_dict)

        # CALL WITHOUT EXPLICIT CLASSES
        pipeline = SelectorPipeline.get_from_configuration(config)()

        assert "DummyPreprocessor1" in [s[0] for s in pipeline.preprocessor.steps]
        assert isinstance(pipeline.selector, DummySelector)

        # Test with converted space
        cs_converted = convert_class_choices_to_categorical(cs)
        config_converted = Configuration(cs_converted, values=config_dict)

        # CALL WITHOUT EXPLICIT CLASSES ON CONVERTED CONFIG
        pipeline_recovered = SelectorPipeline.get_from_configuration(config_converted)()

        assert "DummyPreprocessor1" in [
            s[0] for s in pipeline_recovered.preprocessor.steps
        ]
        assert isinstance(pipeline_recovered.selector, DummySelector)
