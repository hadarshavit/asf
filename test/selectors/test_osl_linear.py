from asf.selectors import OSLLinearSelector


def test_osl_linear_selector(dummy_performance, dummy_features, validate_predictions):
    selector = OSLLinearSelector(budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
    # Check that it has thetas
    assert hasattr(selector, "thetas")
    assert len(selector.thetas) == 2
    for algo in ["algo1", "algo2"]:
        assert algo in selector.thetas
        assert selector.thetas[algo].shape == (2,)  # f1 + intercept
