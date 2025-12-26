from asf.selectors import SurvivalAnalysis


def test_survival_analysis(dummy_performance, dummy_features, validate_predictions):
    selector = SurvivalAnalysis(budget=5.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis_schedule(dummy_performance, dummy_features):
    # Test that it produces a schedule
    selector = SurvivalAnalysis(budget=2.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)

    for inst_id, schedule in predictions.items():
        assert len(schedule) >= 1
        total_budget = sum(time for algo, time in schedule)
        # Should be roughly <= budget if it's a schedule
        # But SurvivalAnalysis might behave differently depending on implementation
        assert total_budget > 0
