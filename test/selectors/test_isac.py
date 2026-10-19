from typing import Any
import pytest

from asf.clustering.wrappers import GMeansWrapper, KMeansWrapper
from asf.selectors import ISAC


def test_isac_selector(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = ISAC()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_isac_tunable_clusterers_support_prediction() -> None:
    params, _, _ = ISAC._define_hyperparameters()
    if not params:
        pytest.skip("ConfigSpace is not installed")
    choices = set(params[0]._class_choices)

    assert choices == {GMeansWrapper, KMeansWrapper}
