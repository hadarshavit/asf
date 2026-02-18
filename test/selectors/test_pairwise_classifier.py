import pandas as pd
import pytest
import numpy as np
from asf.selectors import PairwiseClassifier
from asf.predictors.xgboost import XGBoostClassifierWrapper
from asf.predictors import RegressionMLP, XGBoostRegressorWrapper


def tiny_data():
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    Y = pd.DataFrame({"a": [1.0, 2.0, 0.5], "b": [0.9, 1.5, 2.0]}, index=X.index)
    return X, Y


def test_pairwise_generate_features_and_predict():
    X, Y = tiny_data()
    sel = PairwiseClassifier(model_class=XGBoostClassifierWrapper, budget=3.0)
    sel.fit(X, Y)
    feats = sel.generate_features(X)
    assert list(feats.columns) == sel.algorithms
    preds = sel.predict(X)
    assert isinstance(preds, dict)
    assert set(preds.keys()) == {str(i) for i in X.index}
    for v in preds.values():
        assert isinstance(v, list) and len(v) == 1


def test_pairwise_config_space_roundtrip():
    try:
        import ConfigSpace  # noqa: F401
    except ImportError:
        pytest.skip("ConfigSpace not installed")

    cs = PairwiseClassifier.get_configuration_space()
    assert cs is not None
    # cs_transform is no longer returned by get_configuration_space in ConfigurableMixin


@pytest.mark.parametrize(
    "model_class", [RegressionMLP, XGBoostRegressorWrapper, XGBoostClassifierWrapper]
)
def test_pairwise_classifier(
    dummy_performance, dummy_features, model_class, validate_predictions
):
    # RegressionMLP does not support sample weights, so we disable use_weights
    use_weights = model_class != RegressionMLP
    selector = PairwiseClassifier(model_class=model_class, use_weights=use_weights)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_pairwise_classifier_numpy_output(dummy_performance, dummy_features):
    selector = PairwiseClassifier(prediction_mode="numpy")
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (20, 2)
