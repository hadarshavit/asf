import pytest
from asf.selectors import MetaSelector, SingleBestSolver, SurvivalAnalysis


def test_meta_selector(dummy_performance, dummy_features, validate_predictions):
    # Test MetaSelector with base selectors and a meta_selector
    base_selectors = [SingleBestSolver(budget=3.0), SingleBestSolver(budget=3.0)]
    meta = SingleBestSolver(budget=3.0)
    # Pass them positionally
    selector = MetaSelector(base_selectors, meta, budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_meta_selector_rejects_schedule_base(dummy_performance, dummy_features):
    # MetaSelector currently only supports base selectors that return a single algorithm
    # SurvivalAnalysis returns a schedule if use_schedule=True
    base_selectors = [SurvivalAnalysis(budget=3.0, use_schedule=True)]
    meta = SingleBestSolver(budget=3.0)

    # This should fail during initialization due to RETURN_TYPE check
    with pytest.raises(ValueError, match="must have RETURN_TYPE 'single'"):
        MetaSelector(base_selectors, meta, budget=3.0)
