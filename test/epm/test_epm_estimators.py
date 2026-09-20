import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from asf.epm import EPM


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeRegressor(max_depth=2),
        KNeighborsRegressor(n_neighbors=1),
    ],
)
def test_epm_clones_configured_estimator_on_every_fit(estimator):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = np.array([1.0, 2.0, 3.0])
    epm = EPM(estimator=estimator)
    epm.fit(X, y)
    np.testing.assert_allclose(epm.predict(X), y)
    fitted = epm.predictor
    assert fitted is not estimator
    assert not hasattr(estimator, "n_features_in_")
    epm.fit(X, y)
    assert epm.predictor is not fitted


def test_epm_accepts_legacy_sklearn_class():
    X = np.array([[0.0], [1.0], [2.0]])
    epm = EPM(
        predictor_class=RandomForestRegressor,
        predictor_kwargs={"n_estimators": 2, "random_state": 42},
    )
    epm.fit(X, np.array([1.0, 2.0, 3.0]))
    assert epm.predict(X).shape == (3,)
    assert epm.predictor.model_class.n_estimators == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predictor_kwargs": {"max_depth": 2}},
        {"predictor_config": {}},
    ],
)
def test_epm_rejects_ambiguous_estimator_parameters(kwargs):
    with pytest.raises(ValueError, match="Configure estimator directly"):
        EPM(estimator=DecisionTreeRegressor(), **kwargs)
