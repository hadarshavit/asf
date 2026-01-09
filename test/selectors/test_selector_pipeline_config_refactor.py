from asf.selectors.selector_pipeline import SelectorPipeline
from asf.utils.configurable import ConfigurableMixin
from ConfigSpace import Configuration


# Dummy classes
class DummySelector(ConfigurableMixin):
    PREFIX = "dummy_sel"

    def __init__(self, **kwargs):
        self.budget = 10.0

    @staticmethod
    def _define_hyperparameters(**kwargs):
        return [], [], []


class DummyPreprocessor1(ConfigurableMixin):
    PREFIX = "p1"
    pass


class DummyPreprocessor2(ConfigurableMixin):
    PREFIX = "p2"
    pass


def test_selector_pipeline_config_space_refactor():
    # Define space
    cs = SelectorPipeline.get_configuration_space(
        selector_class=[DummySelector],
        preprocessing_class=[DummyPreprocessor1, DummyPreprocessor2],
    )

    # Check HPs
    hps = dict(cs)

    # Should have preprocessor:DummyPreprocessor1
    hp_name_1 = f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1"
    assert hp_name_1 in hps
    hp1 = hps[hp_name_1]

    # Check choices (ClassChoice converts to Categorical with strings for ConfigSpace)
    # The internal logic maps "False" string to False boolean, but ConfigSpace sees "False"
    assert "DummyPreprocessor1" in hp1.choices  # type: ignore[attr-defined]
    assert "False" in hp1.choices  # type: ignore[attr-defined]

    # Defaults
    assert hp1.default_value == "False"

    # Sample configuration
    # Force enable one
    config_dict = {
        hp_name_1: "DummyPreprocessor1",
        f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor2": "False",
        f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
    }
    config = Configuration(cs, values=config_dict)

    # Create pipeline (no need to pass class lists - auto-discovery from config space)
    pipeline_partial = SelectorPipeline.get_from_configuration(config)
    pipeline = pipeline_partial()

    # Verify preprocessors
    # pipeline.preprocessor is a sklearn Pipeline object wrapped around the list
    # The actual list passed to init is in pipeline.preprocessor.steps
    # steps = [('SimpleImputer', ...), ('DummyPreprocessor1', ...)]
    assert pipeline.preprocessor is not None
    steps = pipeline.preprocessor.steps
    step_names = [s[0] for s in steps]

    assert "SimpleImputer" in step_names[0]
    assert "DummyPreprocessor1" in step_names
    assert "DummyPreprocessor2" not in step_names

    # Verify config retrieval
    cfg = pipeline.get_config()
    # It constructs preprocessor_steps names
    assert "DummyPreprocessor1" in cfg["preprocessor_steps"]


def test_selector_pipeline_none_preprocessing():
    # Test with no preprocessors active
    cs = SelectorPipeline.get_configuration_space(
        selector_class=[DummySelector], preprocessing_class=[DummyPreprocessor1]
    )

    config_dict = {
        f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1": "False",
        f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
    }
    config = Configuration(cs, values=config_dict)

    pipeline = SelectorPipeline.get_from_configuration(config)()

    # Should only have SimpleImputer
    steps = pipeline.preprocessor.steps
    assert len(steps) == 1
    assert steps[0][0] == "SimpleImputer"


def test_selector_pipeline_auto_discovery():
    # Test that we can recover classes from the space without passing them explicitly
    cs = SelectorPipeline.get_configuration_space(
        selector_class=[DummySelector],
        preprocessing_class=[DummyPreprocessor1, DummyPreprocessor2],
    )

    # 1. Test with original space (ClassChoice present)
    config_dict = {
        f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor1": "DummyPreprocessor1",
        f"{SelectorPipeline.PREFIX}:preprocessor:DummyPreprocessor2": "False",
        f"{SelectorPipeline.PREFIX}:selector": "DummySelector",
    }
    config = Configuration(cs, values=config_dict)

    # CALL WITHOUT EXPLICIT CLASSES
    pipeline = SelectorPipeline.get_from_configuration(config)()

    assert "DummyPreprocessor1" in [s[0] for s in pipeline.preprocessor.steps]
    assert isinstance(pipeline.selector, DummySelector)

    # 2. Test with converted space (Categorical + recovered meta)
    from asf.utils.configurable import convert_class_choices_to_categorical

    cs_converted = convert_class_choices_to_categorical(cs)

    config_converted = Configuration(cs_converted, values=config_dict)

    # CALL WITHOUT EXPLICIT CLASSES ON CONVERTED CONFIG
    pipeline_recovered = SelectorPipeline.get_from_configuration(config_converted)()

    assert "DummyPreprocessor1" in [s[0] for s in pipeline_recovered.preprocessor.steps]
    assert isinstance(pipeline_recovered.selector, DummySelector)


if __name__ == "__main__":
    test_selector_pipeline_config_space_refactor()
    test_selector_pipeline_none_preprocessing()
    test_selector_pipeline_auto_discovery()
