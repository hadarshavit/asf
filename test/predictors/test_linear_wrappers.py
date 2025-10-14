import pytest

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace

from asf.predictors.linear_model import (
    LinearClassifierWrapper,
    LinearRegressorWrapper,
)


@pytest.mark.parametrize(
    "wrapper_cls",
    [LinearClassifierWrapper, LinearRegressorWrapper],
)
def test_linear_configuration_space(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)

    prefix = wrapper_cls.PREFIX
    cs_names = {hp.name for hp in cs.get_hyperparameters()}
    assert f"{prefix}:alpha" in cs_names
    assert f"{prefix}:eta0" in cs_names


@pytest.mark.parametrize(
    "wrapper_cls, extra_params",
    [
        (LinearClassifierWrapper, {"loss": "log_loss"}),
        (LinearRegressorWrapper, {"penalty": "l2"}),
    ],
)
def test_linear_get_from_configuration(wrapper_cls, extra_params):
    cs = wrapper_cls.get_configuration_space()
    configuration = cs.get_default_configuration()

    factory = wrapper_cls.get_from_configuration(configuration, **extra_params)
    instance = factory()

    assert isinstance(instance, wrapper_cls)

    prefix = wrapper_cls.PREFIX
    assert instance.model_class.alpha == configuration[f"{prefix}:alpha"]
    assert instance.model_class.eta0 == configuration[f"{prefix}:eta0"]
    for key, value in extra_params.items():
        assert getattr(instance.model_class, key) == value
