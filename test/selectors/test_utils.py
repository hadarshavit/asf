def validate_predictions(predictions):
    """
    Validates that predictions have the expected structure:
    - Length of predictions is 20.
    - Each value in predictions is a list.
    - Each list has a length of 1.
    """
    assert len(predictions) == 20
    for key in predictions:
        assert isinstance(predictions[key], list)
        assert len(predictions[key]) == 1
