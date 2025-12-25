"""Tests for the ConfigurableMixin and ClassChoice utilities."""

from functools import partial


class TestConfigurableMixin:
    """Tests for ConfigurableMixin functionality."""

    def test_class_choice_creation(self):
        """Test that ClassChoice can be created with class choices."""
        from asf.utils.configurable import ClassChoice
        from asf.predictors.random_forest import (
            RandomForestClassifierWrapper,
            RandomForestRegressorWrapper,
        )

        choice = ClassChoice(
            "model",
            choices=[RandomForestClassifierWrapper, RandomForestRegressorWrapper],
        )

        assert choice.name == "model"
        assert len(choice.choices) == 2
        assert "RandomForestClassifierWrapper" in choice._choice_map
        assert "RandomForestRegressorWrapper" in choice._choice_map

    def test_class_choice_get_class(self):
        """Test that get_class returns the correct class."""
        from asf.utils.configurable import ClassChoice
        from asf.predictors.random_forest import (
            RandomForestClassifierWrapper,
            RandomForestRegressorWrapper,
        )

        choice = ClassChoice(
            "model",
            choices=[RandomForestClassifierWrapper, RandomForestRegressorWrapper],
        )

        assert (
            choice.get_class("RandomForestClassifierWrapper")
            is RandomForestClassifierWrapper
        )
        assert (
            choice.get_class("RandomForestRegressorWrapper")
            is RandomForestRegressorWrapper
        )

    def test_class_choice_clone_with_prefix(self):
        """Test that clone_with_prefix creates a properly prefixed copy."""
        from asf.utils.configurable import ClassChoice
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        choice = ClassChoice("model", choices=[RandomForestClassifierWrapper])
        cloned = choice.clone_with_prefix("parent:")

        assert cloned.name == "parent:model"
        assert cloned.choices == choice.choices

    def test_rf_classifier_get_configuration_space_via_mixin(self):
        """Test that RandomForestClassifierWrapper generates correct config space via mixin."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space()

        # Check that hyperparameters are present with correct prefix
        hp_names = [hp.name for hp in cs.get_hyperparameters()]
        assert "rf_classifier:n_estimators" in hp_names
        assert "rf_classifier:min_samples_split" in hp_names
        assert "rf_classifier:min_samples_leaf" in hp_names
        assert "rf_classifier:max_features" in hp_names
        assert "rf_classifier:bootstrap" in hp_names

    def test_rf_classifier_get_configuration_space_with_pre_prefix(self):
        """Test configuration space with a pre_prefix."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space(pre_prefix="parent")

        hp_names = [hp.name for hp in cs.get_hyperparameters()]
        assert "parent:rf_classifier:n_estimators" in hp_names
        assert "parent:rf_classifier:bootstrap" in hp_names

    def test_rf_classifier_get_configuration_space_with_parent_param(self):
        """Test configuration space with parent parameter adds conditions."""
        from ConfigSpace import ConfigurationSpace, Categorical
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = ConfigurationSpace()
        parent = Categorical("selector", items=["rf", "xgb"])
        cs.add(parent)

        RandomForestClassifierWrapper.get_configuration_space(
            cs=cs,
            parent_param=parent,
            parent_value="rf",
        )

        # Check that conditions were added
        conditions = cs.get_conditions()
        assert len(conditions) == 5  # 5 hyperparameters, each with a condition

    def test_rf_classifier_get_from_configuration(self):
        """Test that get_from_configuration returns a partial."""
        from asf.predictors.random_forest import RandomForestClassifierWrapper

        cs = RandomForestClassifierWrapper.get_configuration_space()
        config = cs.get_default_configuration()

        result = RandomForestClassifierWrapper.get_from_configuration(config)

        assert isinstance(result, partial)
        assert result.func is RandomForestClassifierWrapper

        # Instantiate and verify
        instance = result()
        assert isinstance(instance, RandomForestClassifierWrapper)

    def test_mixin_define_hyperparameters_simple(self):
        """Test a simple class using the mixin."""
        from asf.utils.configurable import ConfigurableMixin
        from ConfigSpace import Integer, Categorical

        class SimpleClass(ConfigurableMixin):
            PREFIX = "simple"

            def __init__(self, n_iter: int = 10, verbose: bool = False):
                self.n_iter = n_iter
                self.verbose = verbose

            @staticmethod
            def _define_hyperparameters():
                return (
                    [
                        Integer("n_iter", (1, 100), default=10),
                        Categorical("verbose", items=[True, False], default=False),
                    ],
                    [],
                    [],
                )

        cs = SimpleClass.get_configuration_space()
        hp_names = [hp.name for hp in cs.get_hyperparameters()]
        assert "simple:n_iter" in hp_names
        assert "simple:verbose" in hp_names

        config = cs.get_default_configuration()
        result = SimpleClass.get_from_configuration(config)

        assert isinstance(result, partial)
        instance = result()
        assert instance.n_iter == 10
        assert instance.verbose is False

    def test_mixin_with_conditions(self):
        """Test that conditions are properly cloned with prefixes."""
        from asf.utils.configurable import ConfigurableMixin
        from ConfigSpace import Integer, Categorical, EqualsCondition

        class ConditionalClass(ConfigurableMixin):
            PREFIX = "cond"

            def __init__(self, mode: str = "fast", depth: int = 5):
                self.mode = mode
                self.depth = depth

            @staticmethod
            def _define_hyperparameters():
                mode = Categorical("mode", items=["fast", "deep"], default="fast")
                depth = Integer("depth", (1, 20), default=5)

                # depth is only relevant when mode is "deep"
                conditions = [EqualsCondition(depth, mode, "deep")]

                return [mode, depth], conditions, []

        cs = ConditionalClass.get_configuration_space()

        # Check conditions exist
        conditions = cs.get_conditions()
        assert len(conditions) == 1

        # Check the condition references the prefixed hyperparameters
        cond = conditions[0]
        assert cond.child.name == "cond:depth"
        assert cond.parent.name == "cond:mode"
        assert cond.value == "deep"


class TestClassChoiceNested:
    """Tests for nested configuration spaces with ClassChoice."""

    def test_class_choice_adds_nested_space(self):
        """Test that ClassChoice adds nested configuration spaces."""
        from asf.utils.configurable import ConfigurableMixin, ClassChoice
        from ConfigSpace import Integer

        class ChildA(ConfigurableMixin):
            PREFIX = "child_a"

            def __init__(self, a_param: int = 1):
                self.a_param = a_param

            @staticmethod
            def _define_hyperparameters():
                return [Integer("a_param", (1, 10), default=1)], [], []

        class ChildB(ConfigurableMixin):
            PREFIX = "child_b"

            def __init__(self, b_param: int = 2):
                self.b_param = b_param

            @staticmethod
            def _define_hyperparameters():
                return [Integer("b_param", (1, 10), default=2)], [], []

        class Parent(ConfigurableMixin):
            PREFIX = "parent"

            def __init__(self, child, value: int = 5):
                self.child = child
                self.value = value

            @staticmethod
            def _define_hyperparameters():
                return (
                    [
                        ClassChoice("child", choices=[ChildA, ChildB]),
                        Integer("value", (1, 10), default=5),
                    ],
                    [],
                    [],
                )

        cs = Parent.get_configuration_space()
        hp_names = [hp.name for hp in cs.get_hyperparameters()]

        # Parent's hyperparameters
        assert "parent:child" in hp_names
        assert "parent:value" in hp_names

        # ChildA's hyperparameters (nested under parent:child)
        assert "parent:child:child_a:a_param" in hp_names

        # ChildB's hyperparameters (nested under parent:child)
        assert "parent:child:child_b:b_param" in hp_names

        # Check conditional dependencies
        conditions = cs.get_conditions()
        # At least 2 conditions: one for child_a params, one for child_b params
        assert len(conditions) >= 2

    def test_class_choice_get_from_configuration(self):
        """Test extracting nested classes from configuration."""
        from asf.utils.configurable import ConfigurableMixin, ClassChoice
        from ConfigSpace import Integer

        class ChildA(ConfigurableMixin):
            PREFIX = "child_a"

            def __init__(self, a_param: int = 1):
                self.a_param = a_param

            @staticmethod
            def _define_hyperparameters():
                return [Integer("a_param", (1, 10), default=1)], [], []

        class Parent(ConfigurableMixin):
            PREFIX = "parent"

            def __init__(self, child, value: int = 5):
                self.child = child() if callable(child) else child
                self.value = value

            @staticmethod
            def _define_hyperparameters():
                return (
                    [
                        ClassChoice("child", choices=[ChildA]),
                        Integer("value", (1, 10), default=5),
                    ],
                    [],
                    [],
                )

        cs = Parent.get_configuration_space()
        config = cs.get_default_configuration()

        result = Parent.get_from_configuration(config)
        assert isinstance(result, partial)

        instance = result()
        assert isinstance(instance, Parent)
        assert isinstance(instance.child, ChildA)
        assert instance.child.a_param == 1
        assert instance.value == 5
