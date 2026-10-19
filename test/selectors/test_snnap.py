from typing import Any
from asf.selectors import SNNAP


def test_snnap_selector(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SNNAP()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
