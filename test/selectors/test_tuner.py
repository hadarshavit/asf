import numpy as np
import pandas as pd
import pytest

# Skip entire module if SMAC is not installed
pytest.importorskip("smac")

from asf.selectors import PairwiseClassifier, PairwiseRegressor, tune_selector
from asf.predictors import SVMClassifierWrapper, SVMRegressorWrapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tempfile
import shutil


@pytest.fixture
def dummy_data():
    features = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])  # type: ignore[arg-type]
    performance = pd.DataFrame(
        np.random.exponential(15, (10, 3)),
        columns=["algo1", "algo2", "algo3"],  # type: ignore[arg-type]
    )
    # Feature running time (same shape as features, time to compute each feature for each instance)
    features_running_time = pd.DataFrame(
        np.random.exponential(0.1, (10, 3)),
        columns=["f1", "f2", "f3"],  # type: ignore[arg-type]
    )
    return features, performance, features_running_time


def validate_predictions(predictions, n_instances):
    assert isinstance(predictions, dict)
    assert len(predictions) == n_instances
    for v in predictions.values():
        assert isinstance(v, list)


def test_tune_selector_with_preprocessing(dummy_data):
    """Test tune_selector with preprocessors (no presolvers - they don't support get_from_configuration yet)."""
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
        validate_predictions(predictions, len(features))
    finally:
        shutil.rmtree(output_dir)
