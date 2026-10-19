from typing import Any
import pytest
from asf.selectors import PairwiseRegressor
from asf.predictors import RegressionMLP, XGBoostRegressorWrapper


@pytest.mark.parametrize("model_class", [RegressionMLP, XGBoostRegressorWrapper])
def test_pairwise_regressor(
    dummy_performance: Any,
    dummy_features: Any,
    model_class: Any,
    validate_predictions: Any,
) -> None:
    selector = PairwiseRegressor(model_class=model_class)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
