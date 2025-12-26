import numpy as np
import pandas as pd
import pytest
import shutil
import tempfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Skip entire module if SMAC is not installed
pytest.importorskip("smac")

from asf.selectors import (
    PairwiseClassifier,
    PairwiseRegressor,
    tune_selector,
    SelectorPipeline,
)
from asf.predictors import SVMClassifierWrapper, SVMRegressorWrapper


@pytest.fixture
def dummy_data():
    features = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    performance = pd.DataFrame(
        np.random.exponential(15, (10, 3)),
        columns=["algo1", "algo2", "algo3"],
    )
    # Feature running time
    features_running_time = pd.DataFrame(
        np.random.exponential(0.1, (10, 3)),
        columns=["f1", "f2", "f3"],
    )
    return features, performance, features_running_time


def test_tune_selector_with_preprocessing(dummy_data):
    """Test tune_selector with preprocessors."""
    features, performance, features_running_time = dummy_data
    preprocessors = [StandardScaler, MinMaxScaler]

    output_dir = tempfile.mkdtemp()

    try:
        selector = tune_selector(
            features,
            performance,
            features_running_time=features_running_time,
            selector_class=[
                (PairwiseClassifier, {"model_class": [SVMClassifierWrapper]}),
                (PairwiseRegressor, {"model_class": [SVMRegressorWrapper]}),
            ],
            budget=10.0,
            runcount_limit=2,
            preprocessing_class=preprocessors,
            cv=2,
            seed=42,
            output_dir=output_dir,
        )

        # Check pipeline config includes preprocessor
        config = selector.get_config()
        assert "preprocessor_steps" in config
        assert isinstance(config["preprocessor_steps"], list)
        assert len(config["preprocessor_steps"]) > 0

        # Fit and predict
        selector.fit(features, performance)
        predictions = selector.predict(features)
        assert len(predictions) == len(features)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_selector_tuner(dummy_performance, dummy_features, validate_predictions):
    """Test selector tuner with basic dummy data."""
    # Create dummy feature running time
    features_running_time = pd.DataFrame(
        np.random.exponential(0.1, dummy_features.shape),
        columns=dummy_features.columns,
        index=dummy_features.index,
    )

    output_dir = "./smac_test_output"
    try:
        tuned_pipeline = tune_selector(
            X=dummy_features,
            y=dummy_performance,
            features_running_time=features_running_time,
            selector_class=[PairwiseClassifier, PairwiseRegressor],
            runcount_limit=2,
            cv=2,
            seed=42,
            output_dir=output_dir,
            budget=450.0,
        )
        assert isinstance(tuned_pipeline, SelectorPipeline)
        # Fit the best pipeline found by the tuner
        tuned_pipeline.fit(dummy_features, dummy_performance)
        predictions = tuned_pipeline.predict(dummy_features)
        validate_predictions(predictions)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
