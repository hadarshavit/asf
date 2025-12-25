import os
import shutil

import pytest

from asf.scenario.aslib_reader import read_aslib_scenario
from asf.selectors import PerformanceModel
from asf.selectors.selector_tuner import tune_selector


try:
    import smac  # noqa: F401

    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False


@pytest.fixture(autouse=True)
def cleanup_smac_output():
    """Clean up smac_output directory after each test."""
    yield
    smac_output_dir = os.path.join(os.getcwd(), "smac_output")
    if os.path.exists(smac_output_dir):
        shutil.rmtree(smac_output_dir)


@pytest.fixture()
def scenario_data():
    aslib_path = os.environ.get("ASLIB_PATH")
    if aslib_path is None:
        pytest.skip("ASLIB_PATH environment variable not set")
    assert aslib_path is not None  # for type checker: pytest.skip always raises
    scenario_path = os.path.join(aslib_path, "MAXSAT19-UCMS")
    return read_aslib_scenario(scenario_path)


@pytest.mark.skipif(not SMAC_AVAILABLE, reason="SMAC is not installed")
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


@pytest.mark.skipif(not SMAC_AVAILABLE, reason="SMAC is not installed")
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
        def get_from_configuration(configuration, **kwargs):
            class Inst:
                pass

            from functools import partial

            return partial(Inst)

    config = {"pipeline:selector": DummySelector, "pipeline:max_feature_time": 123.0}

    pipeline = _create_pipeline(
        config,
        1000,
        False,
        {},
        None,
    )

    assert hasattr(pipeline, "max_feature_time")
    assert pipeline.max_feature_time == 123.0
