import importlib.util

import numpy as np
import pytest

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace

from asf.predictors.xgboost import (
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    XGBoostRankerWrapper,
)


XGB_AVAILABLE = importlib.util.find_spec("xgboost") is not None


@pytest.mark.skipif(XGB_AVAILABLE, reason="Only meaningful when xgboost is missing")
def test_xgboost_wrappers_require_dependency():
    with pytest.raises(ImportError):
        XGBoostClassifierWrapper()


@pytest.mark.skipif(not XGB_AVAILABLE, reason="Requires xgboost optional dependency")
@pytest.mark.parametrize(
    "wrapper_cls",
    [XGBoostClassifierWrapper, XGBoostRegressorWrapper, XGBoostRankerWrapper],
)
def test_xgboost_configuration_space(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)

    prefix = wrapper_cls.PREFIX
    cs_names = {hp.name for hp in cs.get_hyperparameters()}
    assert {f"{prefix}:max_depth", f"{prefix}:lambda"}.issubset(cs_names)


@pytest.mark.skipif(not XGB_AVAILABLE, reason="Requires xgboost optional dependency")
@pytest.mark.parametrize(
    "wrapper_cls,data_fixture,eval_metric",
    [
        (XGBoostClassifierWrapper, "classification_data", "logloss"),
        (XGBoostRegressorWrapper, "regression_data", "rmse"),
    ],
)
def test_xgboost_fit_predict(wrapper_cls, data_fixture, eval_metric, request):
    X, y = request.getfixturevalue(data_fixture)

    if wrapper_cls is XGBoostClassifierWrapper:
        y = y.astype(bool)

    factory = wrapper_cls.get_from_configuration(
        wrapper_cls.get_configuration_space().get_default_configuration(),
        n_estimators=10,
        use_label_encoder=False,
        eval_metric=eval_metric,
    )
    instance = factory()

    instance.fit(X, y)
    predictions = instance.predict(X)

    assert predictions.shape[0] == X.shape[0]
    if wrapper_cls is XGBoostClassifierWrapper:
        assert predictions.dtype == np.bool_
