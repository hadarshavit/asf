import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace

from asf.predictors.mlp import MLPClassifierWrapper, MLPRegressorWrapper


@pytest.mark.parametrize(
    "wrapper_cls",
    [MLPClassifierWrapper, MLPRegressorWrapper],
)
def test_mlp_configuration_space(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)

    prefix = wrapper_cls.PREFIX
    cs_names = {hp.name for hp in cs.get_hyperparameters()}
    assert {f"{prefix}:depth", f"{prefix}:width"}.issubset(cs_names)


@pytest.mark.parametrize(
    "wrapper_cls",
    [MLPClassifierWrapper, MLPRegressorWrapper],
)
def test_mlp_get_from_configuration(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    configuration = cs.get_default_configuration()

    factory = wrapper_cls.get_from_configuration(configuration, max_iter=32)
    instance = factory()

    assert isinstance(instance, wrapper_cls)

    prefix = wrapper_cls.PREFIX
    depth = configuration[f"{prefix}:depth"]
    width = configuration[f"{prefix}:width"]
    assert instance.model_class.hidden_layer_sizes == tuple([width] * depth)
    assert instance.model_class.max_iter == 32


def test_mlp_classifier_fit_asserts_on_sample_weight(classification_data):
    X, y = classification_data
    wrapper = MLPClassifierWrapper({"hidden_layer_sizes": (4,), "max_iter": 32})

    with pytest.raises(AssertionError):
        wrapper.fit(X, y, sample_weight=np.ones(len(y)))


def test_mlp_regressor_fit_asserts_on_sample_weight(regression_data):
    X, y = regression_data
    wrapper = MLPRegressorWrapper({"hidden_layer_sizes": (4,), "max_iter": 32})

    with pytest.raises(AssertionError):
        wrapper.fit(X, y, sample_weight=np.ones(len(y)))


@pytest.mark.parametrize(
    "wrapper_cls, data_fixture",
    [
        (MLPClassifierWrapper, "classification_data"),
        (MLPRegressorWrapper, "regression_data"),
    ],
)
def test_mlp_fit_predict(wrapper_cls, data_fixture, request):
    X, y = request.getfixturevalue(data_fixture)
    wrapper = wrapper_cls({"hidden_layer_sizes": (4,), "max_iter": 64})

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        wrapper.fit(X, y)

    predictions = wrapper.predict(X)
    assert predictions.shape[0] == X.shape[0]
