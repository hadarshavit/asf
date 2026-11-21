from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from asf.selectors import SimpleRanking, JointRanking


def test_simple_ranking(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SimpleRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


@pytest.mark.parametrize(
    "maximize,labels,scores,expected",
    [
        (False, [3.0, 2.0, 1.0], [0.9, 0.5, 0.1], "low"),
        (True, [1.0, 2.0, 3.0], [0.1, 0.5, 0.9], "high"),
    ],
)
def test_simple_ranking_relevance_direction(
    maximize: bool, labels: list[float], scores: list[float], expected: str
) -> None:
    features = pd.DataFrame({"f": [0.0]}, index=["instance"])
    performance = pd.DataFrame(
        {"low": [1.0], "middle": [5.0], "high": [10.0]}, index=features.index
    )
    ranker = Mock()
    ranker.predict.return_value = np.array(scores)
    selector = SimpleRanking(model_class=lambda: ranker, maximize=maximize)
    selector.fit(features, performance)

    assert ranker.fit.call_args.args[1].tolist() == labels
    assert selector.predict(features) == {"instance": [(expected, 0.0)]}


def test_joint_ranking(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = JointRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
