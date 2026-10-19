from typing import Any
from asf.selectors import ISA


def test_isa_selector(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    # Use a small budget so not all instances are solved by all algorithms
    selector = ISA(budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
