import pytest

from asf.clustering.wrappers import GMeansWrapper, KMeansWrapper
from asf.selectors import ISAC


def test_isac_selector(dummy_performance, dummy_features, validate_predictions):
    selector = ISAC()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_isac_tunable_clusterers_support_prediction():
    params, _, _ = ISAC._define_hyperparameters()
    if not params:
        pytest.skip("ConfigSpace is not installed")
    choices = set(params[0]._class_choices)

    assert choices == {GMeansWrapper, KMeansWrapper}
