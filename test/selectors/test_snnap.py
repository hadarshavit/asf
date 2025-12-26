from asf.selectors import SNNAP


def test_snnap_selector(dummy_performance, dummy_features, validate_predictions):
    selector = SNNAP()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
