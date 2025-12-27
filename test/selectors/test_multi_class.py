import pytest
from asf.selectors import MultiClassClassifier
from asf.predictors import RegressionMLP, XGBoostClassifierWrapper


@pytest.mark.parametrize("model_class", [RegressionMLP, XGBoostClassifierWrapper])
def test_multi_class_classifier(
    dummy_performance, dummy_features, model_class, validate_predictions
):
    selector = MultiClassClassifier(model_class=model_class)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
