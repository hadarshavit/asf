import numpy as np
import pandas as pd
import pytest
from asf.selectors import PairwiseClassifier, PairwiseRegressor, tune_selector
from asf.predictors import SVMClassifierWrapper, SVMRegressorWrapper
from asf.presolving.asap_v2 import ASAPv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@pytest.fixture
def dummy_data():
    features = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    performance = pd.DataFrame(
        np.random.exponential(15, (10, 3)), columns=["algo1", "algo2", "algo3"]
    )
    return features, performance


def validate_predictions(predictions, n_instances):
    assert isinstance(predictions, dict)
    assert len(predictions) == n_instances
    for v in predictions.values():
        assert isinstance(v, list)


def test_tune_selector_with_presolving_and_preprocessing(dummy_data):
    features, performance = dummy_data
    preprocessors = [StandardScaler(), MinMaxScaler()]
    presolvers = [ASAPv2()]

    selector = tune_selector(
        features,
        performance,
        selector_class=[
            (PairwiseClassifier, {"model_class": [SVMClassifierWrapper]}),
            (PairwiseRegressor, {"model_class": [SVMRegressorWrapper]}),
        ],
        budget=10.0,
        runcount_limit=2,
        preprocessing_class=preprocessors,
        pre_solving_class=presolvers,
        cv=2,
        seed=42,
    )

    # Check pipeline config includes preprocessor and presolver
    config = selector.get_config()
    assert "preprocessor_steps" in config
    assert isinstance(config["preprocessor_steps"], list)
    assert len(config["preprocessor_steps"]) > 0
    assert "pre_solving" in config
    assert config["pre_solving"] in ["ASAPv2"]

    # Fit and predict
    selector.fit(features, performance)
    predictions = selector.predict(features)
    validate_predictions(predictions, len(features))
