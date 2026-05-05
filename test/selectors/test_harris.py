import pytest
from asf.selectors.hybrid_decision_tree import HARRIS


@pytest.mark.parametrize(
    "lambda_param,max_features",
    [
        (0.0, "sqrt"),
        (0.5, "log2"),
        (1.0, 3),
    ],
)
def test_harris_selector_basic(
    dummy_performance, dummy_features, validate_predictions, lambda_param, max_features
):
    """Test HARRIS with 3 cases: lambda in {0,0.5,1} and max_features in {sqrt,log2,3}."""
    selector = HARRIS(
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        lambda_param=lambda_param,
        max_features=max_features,
        max_thresholds=5,
        budget=450.0,
        random_state=42,
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_harris_selector_refit_resets_forest(dummy_performance, dummy_features):
    selector = HARRIS(n_estimators=3, max_thresholds=5)
    selector.fit(dummy_features, dummy_performance)
    selector.fit(dummy_features, dummy_performance)

    assert len(selector.trees) == 3
    assert len(selector.feature_indices_per_tree) == 3
