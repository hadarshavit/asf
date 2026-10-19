from typing import Any
from asf.selectors import SimpleRanking, JointRanking


def test_simple_ranking(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SimpleRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_joint_ranking(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = JointRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
