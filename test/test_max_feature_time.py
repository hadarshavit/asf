import pytest

from asf.scenario.aslib_reader import read_aslib_scenario
from asf.selectors import PerformanceModel
from asf.selectors.selector_tuner import tune_selector


@pytest.fixture()
def scenario_data():
    scenario_path = "/home/ni574034/asf/paper/aslib_data/MAXSAT19-UCMS"
    return read_aslib_scenario(scenario_path)


def test_max_feature_time_is_tunable_and_applied(scenario_data):
    (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    ) = scenario_data

    # Run a very short HPO (runcount_limit=1) that allows SMAC to expose a value
    sel = tune_selector(
        X=features,
        y=performance,
        selector_class=PerformanceModel,
        features_running_time=features_running_time,
        runcount_limit=1,
        cv=2,
        budget=budget,
        max_feature_time=None,
        smac_kwargs=lambda s: {"overwrite": True, "logging_level": False},
    )

    # SMAC should have provided a value for the cap and pipeline should expose it
    assert hasattr(sel, "max_feature_time")
    assert sel.max_feature_time is not None
    assert isinstance(sel.max_feature_time, (int, float))
    assert sel.max_feature_time >= 0.0
    assert sel.max_feature_time <= float(budget)


def test_fixed_max_feature_time_is_respected(scenario_data):
    (
        features,
        performance,
        features_running_time,
        cv,
        feature_groups,
        maximize,
        budget,
        algorithm_features,
    ) = scenario_data

    fixed = 42.0
    sel = tune_selector(
        X=features,
        y=performance,
        selector_class=PerformanceModel,
        features_running_time=features_running_time,
        runcount_limit=1,
        cv=2,
        budget=budget,
        max_feature_time=fixed,
        smac_kwargs=lambda s: {"overwrite": True, "logging_level": False},
    )

    assert hasattr(sel, "max_feature_time")
    assert sel.max_feature_time == fixed


def test__create_pipeline_prefers_config_value():
    # This test does not require ConfigSpace or scenario data.
    from asf.selectors.selector_tuner import _create_pipeline

    class DummySelector:
        @staticmethod
        def get_from_configuration(config, cs_transform, **kwargs):
            class Inst:
                pass

            return Inst()

    config = {"selector": "Dummy", "max_feature_time": 123.0}
    cs_transform = {"selector": {"Dummy": DummySelector}}

    pipeline = _create_pipeline(
        config=config,
        cs_transform=cs_transform,
        budget=1000,
        maximize=False,
        selector_kwargs={},
        feature_selector=None,
        algorithm_pre_selector=None,
        max_feature_time=None,
    )

    assert hasattr(pipeline, "max_feature_time")
    assert pipeline.max_feature_time == 123.0
