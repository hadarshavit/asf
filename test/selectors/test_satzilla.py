from typing import Any
from asf.selectors import SATzilla


def test_satzilla_selector(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SATzilla(budget=3600.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert isinstance(predictions, dict)
    validate_predictions(predictions)
    # Check that it has epms
    assert len(selector.epms) >= 2
    for inst_id, schedule in predictions.items():
        assert isinstance(schedule, list)
        assert len(schedule) > 0
