import numpy as np
import pandas as pd

from asf.selectors.performance_model import PerformanceModel


from asf.predictors.abstract_predictor import AbstractPredictor


class DummyRegressor(AbstractPredictor):
    def __init__(self, **kwargs):
        self._y = None
        self._X_cols = None

    def fit(self, X, y):
        # X is DataFrame, y is Series or DataFrame
        self._X_cols = list(X.columns)
        # Learn column-wise means if 2D, else scalar mean
        if hasattr(y, "shape") and len(getattr(y, "shape", ())) == 2:
            self._y = np.asarray(y).mean(axis=0)
        else:
            self._y = float(np.asarray(y).mean())

    def predict(self, X):
        n = len(X)
        if isinstance(self._y, float):
            return np.full(n, self._y)
        else:
            # multi-target prediction
            return np.tile(self._y, (n, 1))

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


def small_df():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
    Y = pd.DataFrame({"a": [3.0, 1.0, 2.0], "b": [2.0, 3.0, 1.0]}, index=X.index)
    return X, Y


def test_performance_model_single_target_branch():
    X, Y = small_df()
    pm = PerformanceModel(model_class=DummyRegressor, budget=5.0, normalize=None)
    pm.fit(X, Y)
    preds = pm.predict(X)
    # Should choose algorithm with min predicted mean per-row; Dummy predicts per-algo means -> tie-breaking by argmin
    assert set(preds.keys()) == set(X.index)
    for lst in preds.values():
        algo, bud = lst[0]
        assert algo in ("a", "b")
        assert bud == 5.0


def test_performance_model_multi_target_only():
    X, Y = small_df()

    # Case 1: use_multi_target=True -> fit single model that outputs 2 targets
    pm_mt = PerformanceModel(
        model_class=DummyRegressor, use_multi_target=True, normalize=None, budget=7.0
    )
    pm_mt.fit(X, Y)
    preds_mt = pm_mt.predict(X)
    assert set(preds_mt.keys()) == set(X.index)
