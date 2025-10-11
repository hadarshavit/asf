import pandas as pd
import numpy as np

from asf.epm.epm import EPM


class DummyPredictor:
    """A tiny predictor that stores mean of y and predicts it for any X."""

    def __init__(self, **kwargs):
        self._mean = None

    def fit(self, X, y, sample_weight=None):
        # y might be a pandas Series or numpy array
        self._mean = float(np.mean(np.asarray(y)))

    def predict(self, X):
        # return the mean for each row in X
        n = len(X)
        return np.array([self._mean] * n)


class IdentityNorm:
    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def inverse_transform(self, X):
        return np.asarray(X)


def test_epm_with_dummy_predictor_numpy_input():
    # Create X as numpy array and y as numpy array
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([10.0, 20.0])

    model = EPM(
        predictor_class=DummyPredictor,
        normalization_class=IdentityNorm,
        transform_back=False,
    )
    model.fit(X, y)
    preds = model.predict(X)
    # Dummy predictor predicts mean of y (without normalization)
    assert np.allclose(preds, np.array([15.0, 15.0]))


def test_epm_transform_back_uses_normalization():
    # Ensure that when transform_back is True, inverse_transform is applied
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([100.0, 400.0])

    # Use IdentityNorm so normalization does nothing
    model = EPM(
        predictor_class=DummyPredictor,
        normalization_class=IdentityNorm,
        transform_back=True,
    )
    model.fit(X, y)
    p = model.predict(X)
    # Dummy predictor predicts mean of y
    assert np.allclose(p, np.array([250.0, 250.0]))
