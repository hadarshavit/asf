from asf.selectors import CollaborativeFilteringSelector


def test_collaborative_filtering_selector(
    dummy_performance, dummy_features, validate_predictions
):
    selector = CollaborativeFilteringSelector()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
    # Check that it has the expected attributes
    assert hasattr(selector, "performance_matrix")
    assert selector.performance_matrix is not None
    assert selector.performance_matrix.shape == (20, 2)
