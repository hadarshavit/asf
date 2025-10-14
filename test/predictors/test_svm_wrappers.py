import pytest

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace

from asf.predictors.svm import SVMClassifierWrapper, SVMRegressorWrapper


@pytest.mark.parametrize(
    "wrapper_cls",
    [SVMClassifierWrapper, SVMRegressorWrapper],
)
def test_svm_configuration_space(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    assert isinstance(cs, ConfigurationSpace)

    prefix = wrapper_cls.PREFIX
    cs_names = {hp.name for hp in cs.get_hyperparameters()}
    assert {f"{prefix}:kernel", f"{prefix}:C"}.issubset(cs_names)


@pytest.mark.parametrize(
    "wrapper_cls, extra",
    [
        (SVMClassifierWrapper, {"probability": True}),
        (SVMRegressorWrapper, {"epsilon": 0.2}),
    ],
)
def test_svm_get_from_configuration(wrapper_cls, extra):
    cs = wrapper_cls.get_configuration_space()
    configuration = cs.get_default_configuration()

    factory = wrapper_cls.get_from_configuration(configuration, **extra)
    instance = factory()

    assert isinstance(instance, wrapper_cls)

    prefix = wrapper_cls.PREFIX
    assert instance.model_class.kernel == configuration[f"{prefix}:kernel"]
    assert instance.model_class.C == configuration[f"{prefix}:C"]
    for key, value in extra.items():
        assert getattr(instance.model_class, key) == value


@pytest.mark.parametrize(
    "wrapper_cls, data_fixture",
    [
        (SVMClassifierWrapper, "classification_data"),
        (SVMRegressorWrapper, "regression_data"),
    ],
)
def test_svm_fit_predict(wrapper_cls, data_fixture, request):
    X, y = request.getfixturevalue(data_fixture)
    cs = wrapper_cls.get_configuration_space()
    configuration = cs.get_default_configuration()

    factory = wrapper_cls.get_from_configuration(configuration)
    instance = factory()
    instance.fit(X, y)

    predictions = instance.predict(X)
    assert predictions.shape[0] == X.shape[0]
