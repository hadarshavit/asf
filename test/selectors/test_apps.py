from typing import cast

from asf.selectors.parallel_portfolio_selector import APPS
from asf.predictors.random_forest import RandomForestRegressorWrapper
import numpy as np


def test_apps_selector(dummy_performance, dummy_features):
    """Test the APPS (Automatic Parallel Portfolio Selector)."""
    selector = APPS(
        model_class=RandomForestRegressorWrapper,
        p_intersection=0.1,
        n_estimators_for_std=3,
        random_state=42,
    )

    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert isinstance(predictions, dict)
    predictions = cast(dict[str, list[tuple[str, float]]], predictions)

    assert len(predictions) == len(dummy_features), "Predictions length mismatch"
    assert all(isinstance(v, list) for v in predictions.values()), (
        "Not all predictions are lists"
    )

    for inst_name, portfolio in predictions.items():
        assert isinstance(portfolio, list)
        assert len(portfolio) > 0, f"Empty portfolio for {inst_name}"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in portfolio), (
            f"Portfolio for {inst_name} should contain (algo, budget) tuples"
        )
        assert all(
            isinstance(algo, str) and isinstance(budget, float)
            for algo, budget in portfolio
        ), f"Portfolio for {inst_name} tuples should be (str, float)"
        assert all(algo in ["algo1", "algo2", "algo3"] for algo, _ in portfolio), (
            f"Portfolio for {inst_name} contains invalid algorithm names"
        )


def test_apps_selector_different_thresholds(dummy_performance, dummy_features):
    """Test that different p_intersection values produce different portfolio sizes."""
    portfolios_by_p = {}

    for p_val in [0.01, 0.5]:
        selector = APPS(
            model_class=RandomForestRegressorWrapper,
            p_intersection=p_val,
            n_estimators_for_std=3,
            random_state=42,
        )
        selector.fit(dummy_features, dummy_performance)
        predictions = selector.predict(dummy_features)
        assert isinstance(predictions, dict)
        predictions = cast(dict[str, list[tuple[str, float]]], predictions)

        avg_size = np.mean([len(portfolio) for portfolio in predictions.values()])
        portfolios_by_p[p_val] = avg_size

    # Lower p_intersection should result in larger portfolios
    assert portfolios_by_p[0.01] > portfolios_by_p[0.5], (
        "Lower p_intersection should produce larger portfolios"
    )
