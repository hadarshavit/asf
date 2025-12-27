from asf.selectors import ISAC


def test_isac_selector(dummy_performance, dummy_features, validate_predictions):
    selector = ISAC()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
