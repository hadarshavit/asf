import pytest

pytest.importorskip("ConfigSpace")
from ConfigSpace import ConfigurationSpace

from asf.predictors.random_forest import (
    RandomForestClassifierWrapper,
    RandomForestRegressorWrapper,
)


@pytest.mark.parametrize(
    "wrapper_cls",
    [RandomForestClassifierWrapper, RandomForestRegressorWrapper],
)
def test_random_forest_configuration_space(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()

    assert isinstance(cs, ConfigurationSpace)

    prefix = wrapper_cls.PREFIX
    # Note: ConfigurableMixin uses colon separator for better readability
    expected_names = {
        f"{prefix}:n_estimators",
        f"{prefix}:min_samples_split",
        f"{prefix}:min_samples_leaf",
        f"{prefix}:max_features",
        f"{prefix}:bootstrap",
    }

    cs_names = {hp.name for hp in list(cs.values())}
    assert expected_names.issubset(cs_names)


@pytest.mark.parametrize(
    "wrapper_cls",
    [RandomForestClassifierWrapper, RandomForestRegressorWrapper],
)
def test_random_forest_get_from_configuration(wrapper_cls):
    cs = wrapper_cls.get_configuration_space()
    configuration = cs.get_default_configuration()

    factory = wrapper_cls.get_from_configuration(configuration)
    instance = factory()

    assert isinstance(instance, wrapper_cls)

    prefix = wrapper_cls.PREFIX
    # Note: ConfigurableMixin uses colon separator for better readability
    assert instance.model_class.n_estimators == configuration[f"{prefix}:n_estimators"]
    assert (
        instance.model_class.min_samples_split
        == configuration[f"{prefix}:min_samples_split"]
    )
