import numpy as np
import pandas as pd
import pytest
from typing import Any
from asf.selectors import PerformanceModel
from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors import RegressionMLP, XGBoostRegressorWrapper


class DummyRegressor(AbstractPredictor):
    def __init__(self, **kwargs):
        self._y = None
        self._X_cols = None

    def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
        self._X_cols = list(X.columns)
        if hasattr(Y, "shape") and len(getattr(Y, "shape", ())) == 2:
            self._y = np.asarray(Y).mean(axis=0)
        else:
            self._y = float(np.asarray(Y).mean())

    def predict(self, X: Any, **kwargs: Any) -> Any:
        n = len(X)
        if isinstance(self._y, float) or (
            hasattr(self._y, "ndim") and self._y.ndim == 0
        ):
            return np.full(n, self._y)
        else:
            return np.tile(self._y, (n, 1))

    def save(self, file_path: str) -> None:
        pass

    @classmethod
    def load(cls, file_path: str) -> "AbstractPredictor":
        return cls()


def small_df():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=pd.Index(["i1", "i2", "i3"]))
    Y = pd.DataFrame(
        {"a": [3.0, 1.0, 2.0], "b": [2.0, 3.0, 1.0]}, index=pd.Index(X.index)
    )
    return X, Y


def test_performance_model_single_target_branch():
    X, Y = small_df()
    pm = PerformanceModel(model_class=DummyRegressor, budget=5.0, normalize=None)
    pm.fit(X, Y)
    preds = pm.predict(X)
    assert set(preds.keys()) == set(X.index)
    for lst in preds.values():
        algo, bud = lst[0]
        assert algo in ("a", "b")
        assert bud == 5.0


def test_performance_model_multi_target_only():
    X, Y = small_df()
    pm_mt = PerformanceModel(
        model_class=DummyRegressor, use_multi_target=True, normalize=None, budget=7.0
    )
    pm_mt.fit(X, Y)
    preds_mt = pm_mt.predict(X)
    assert set(preds_mt.keys()) == set(X.index)


@pytest.mark.parametrize("model_class", [RegressionMLP, XGBoostRegressorWrapper])
def test_performance_model(
    dummy_performance, dummy_features, model_class, validate_predictions
):
    selector = PerformanceModel(model_class=model_class)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_performance_model_with_normalization():
    from asf.preprocessing.performance_scaling import MinMaxNormalization

    X, Y = small_df()
    # Ensure multi-column
    assert Y.shape[1] > 1

    selector = PerformanceModel(
        model_class=DummyRegressor, normalize=MinMaxNormalization()
    )
    # This should not raise ValueError
    selector.fit(X, Y)
    predictions = selector.predict(X)
    assert set(predictions.keys()) == set(X.index)
