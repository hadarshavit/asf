from typing import Any
from asf.selectors import CSHCSelector, SingleBestSolver


def test_cshc_selector_with_backup(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    primary = SingleBestSolver(budget=3.0)
    backup = SingleBestSolver(budget=3.0)
    selector = CSHCSelector(primary, backup, budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
    assert selector.backup_selector is not None


def test_cshc_selector_no_backup(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    # It still needs a primary_selector
    primary = SingleBestSolver(budget=3.0)
    selector = CSHCSelector(primary, budget=3.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)
