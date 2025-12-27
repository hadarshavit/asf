import pytest
from asf.selectors import PairwiseRegressor
from asf.predictors import RegressionMLP, XGBoostRegressorWrapper


@pytest.mark.parametrize("model_class", [RegressionMLP, XGBoostRegressorWrapper])
def test_pairwise_regressor(
    dummy_performance, dummy_features, model_class, validate_predictions
):
    selector = PairwiseRegressor(model_class=model_class)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
