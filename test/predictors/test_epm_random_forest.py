import numpy as np
import pytest

from asf.predictors.epm_random_forest import EPMRandomForest as ForestRegressorWrapper
from asf.predictors.epm_extra_trees import EPMExtraTrees as ExtraTreesWrapper


@pytest.mark.parametrize(
    "wrapper_cls",
    [ForestRegressorWrapper, ExtraTreesWrapper],
)
def test_epm_random_forest_rejects_sample_weight(wrapper_cls, regression_data):
    X, y = regression_data
    model = wrapper_cls(n_estimators=4, random_state=0)

    with pytest.raises(AssertionError):
        model.fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.parametrize(
    "wrapper_cls",
    [ForestRegressorWrapper, ExtraTreesWrapper],
)
def test_epm_random_forest_predict_shapes(wrapper_cls, regression_data):
    X, y = regression_data
    model = wrapper_cls(n_estimators=8, random_state=0)
    model.fit(X, y)

    predictions = model.predict(X)

    if wrapper_cls is ForestRegressorWrapper:
        means, vars = (
            predictions if isinstance(predictions, tuple) else (predictions, None)
        )
        assert means.shape[0] == X.shape[0]
        if vars is not None:
            assert vars.shape[0] == X.shape[0]
    else:
        means, vars = predictions
        assert means.shape == (X.shape[0], 1)
        assert vars.shape == (X.shape[0], 1)
        assert np.all(vars >= 0)


def test_epm_random_forest_return_var(regression_data):
    X, y = regression_data
    model = ForestRegressorWrapper(n_estimators=6, random_state=0, return_var=True)
    model.fit(X, y)

    means, vars = model.predict(X)
    assert means.shape[0] == X.shape[0]
    assert vars.shape[0] == X.shape[0]
    assert np.all(vars >= 0)
