from asf.selectors import SimpleRanking, JointRanking


def test_simple_ranking(dummy_performance, dummy_features, validate_predictions):
    selector = SimpleRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_joint_ranking(dummy_performance, dummy_features, validate_predictions):
    selector = JointRanking()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
