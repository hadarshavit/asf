"""
ConfigurableMixin: Simplify configuration space management for configurable classes.

This module provides:
- ClassChoice: A Categorical hyperparameter that stores actual class references
- ConfigurableMixin: A mixin that automates get_configuration_space and get_from_configuration
"""

from __future__ import annotations

from functools import partial
from typing import Any

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter

    CONFIGSPACE_AVAILABLE = True

    class ClassChoice(CategoricalHyperparameter):
        """
        A Categorical hyperparameter that allows classes as choices.

        This eliminates the need for cs_transform by storing the actual class
        references and using class names as the categorical items.

        Example:
            model_class = ClassChoice(
                "model_class",
                choices=[RandomForestWrapper, XGBoostWrapper],
            )
        """

        def __init__(
            self,
            name: str,
            choices: list[type | bool],
            default: type | bool | None = None,
            weights: tuple[float, ...] | None = None,
        ):
            """
            Initialize a ClassChoice hyperparameter.

            Parameters
            ----------
            name : str
                Name of the hyperparameter (will be prefixed by the mixin).
            choices : list[type | bool]
                List of classes that can be chosen. Each class should implement
                ConfigurableMixin if it has its own hyperparameters. Can also include False.
            default : type, bool, or None, optional
                Default choice. If None, the first choice is used.
            weights : tuple[float, ...] or None, optional
                Weights for sampling, passed to CategoricalHyperparameter.
            """
            self._class_choices = choices  # Store actual choice list
            self._choice_map = {}
            items = []

            for choice in choices:
                if choice is False:
                    self._choice_map["False"] = False
                    items.append("False")
                else:
                    self._choice_map[choice.__name__] = choice
                    items.append(choice.__name__)

            if default is False:
                default_value = "False"
            elif default is not None:
                default_value = default.__name__
            else:
                default_value = items[0]

            super().__init__(
                name, choices=tuple(items), default_value=default_value, weights=weights
            )

        def get_class(self, name: str) -> type:
            """Get the class corresponding to a class name."""
            return self._choice_map[name]

        def to_categorical(self) -> Categorical:
            """
            Convert to a regular Categorical hyperparameter for SMAC serialization.

            This is needed because SMAC only knows how to serialize built-in
            ConfigSpace types, not custom subclasses like ClassChoice.
            """
            weights = None
            if hasattr(self, "probabilities") and self.probabilities is not None:
                weights = tuple(self.probabilities)

            # Store actual class choices in meta to allow recovery after conversion
            meta = (self.meta if hasattr(self, "meta") else {}) or {}
            meta = meta.copy()
            meta["__class_choices__"] = self._class_choices

            return Categorical(
                name=self.name,
                items=list(self.choices),
                default=self.default_value,
                weights=weights,
                meta=meta,
            )

        def clone_with_prefix(self, prefix: str) -> "ClassChoice":
            """Create a copy of this hyperparameter with a prefixed name."""
            new_name = f"{prefix}{self.name}" if prefix else self.name
            # Use stored _class_choices (actual classes) not inherited choices (strings)
            # Convert probabilities to tuple if it exists (stored as numpy array)
            weights = None
            if hasattr(self, "probabilities") and self.probabilities is not None:
                weights = tuple(self.probabilities)
            # Retrieve the default value object from the map using the string representation
            default_obj = self._choice_map.get(self.default_value)

            cloned = ClassChoice(
                name=new_name,
                choices=self._class_choices,
                default=default_obj,
                weights=weights,
            )
            return cloned

except ImportError:
    CONFIGSPACE_AVAILABLE = False
    Hyperparameter = Any  # type: ignore

    # Provide a dummy ClassChoice for type hints when ConfigSpace is not installed
    class ClassChoice:  # type: ignore
        """Dummy ClassChoice when ConfigSpace is not installed."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ConfigSpace is not installed. Install with: pip install 'asf[configspace]'"
            )


def convert_class_choices_to_categorical(cs: ConfigurationSpace) -> ConfigurationSpace:
    """
    Convert all ClassChoice hyperparameters in a ConfigurationSpace to regular Categorical.

    This is needed for SMAC serialization compatibility.

    Parameters
    ----------
    cs : ConfigurationSpace
        The configuration space potentially containing ClassChoice hyperparameters.

    Returns
    -------
    ConfigurationSpace
        A new configuration space with ClassChoice replaced by Categorical.
    """
    if not CONFIGSPACE_AVAILABLE:
        return cs

    # Check if there are any ClassChoice hyperparameters
    class_choices = [hp for hp in cs.values() if isinstance(hp, ClassChoice)]
    if not class_choices:
        return cs  # No conversion needed

    # Create a new ConfigurationSpace with converted hyperparameters
    new_cs = ConfigurationSpace()
    hp_map = {}  # Map old names to new hyperparameters

    # First pass: convert all hyperparameters
    for hp in cs.values():
        if isinstance(hp, ClassChoice):
            new_hp = hp.to_categorical()
        else:
            new_hp = hp  # Keep as-is
        hp_map[hp.name] = new_hp
        new_cs.add(new_hp)

    # Second pass: re-add conditions with updated references
    for condition in cs.conditions:
        # Create new condition with updated hyperparameter references
        child = hp_map[condition.child.name]
        parent = hp_map[condition.parent.name]
        new_condition = type(condition)(
            child=child, parent=parent, value=condition.value
        )
        new_cs.add(new_condition)

    # Third pass: re-add forbidden clauses
    for forbidden in cs.forbidden_clauses:
        new_cs.add(forbidden)

    return new_cs


def _clone_hyperparameter(hp: Hyperparameter, prefix: str) -> Hyperparameter:
    """
    Clone a hyperparameter with a new prefixed name.

    Parameters
    ----------
    hp : Hyperparameter
        The hyperparameter to clone.
    prefix : str
        The prefix to add to the hyperparameter name.

    Returns
    -------
    Hyperparameter
        A cloned hyperparameter with the prefixed name.
    """
    if isinstance(hp, ClassChoice):
        return hp.clone_with_prefix(prefix)

    new_name = f"{prefix}{hp.name}" if prefix else hp.name

    # Get hyperparameter meta information
    meta = hp.meta if hasattr(hp, "meta") else None

    # Handle different hyperparameter types - use actual classes not factory functions
    from ConfigSpace.hyperparameters import (
        IntegerHyperparameter,
        FloatHyperparameter,
        CategoricalHyperparameter,
        OrdinalHyperparameter as OrdinalHP,
        Constant as ConstantHP,
    )
    from ConfigSpace import Integer, Float, Categorical, Constant

    if isinstance(hp, IntegerHyperparameter):
        return Integer(
            name=new_name,
            bounds=(hp.lower, hp.upper),
            default=hp.default_value,
            log=hp.log,
            meta=meta,
        )
    elif isinstance(hp, FloatHyperparameter):
        return Float(
            name=new_name,
            bounds=(hp.lower, hp.upper),
            default=hp.default_value,
            log=hp.log,
            meta=meta,
        )
    elif isinstance(hp, CategoricalHyperparameter):
        # Convert probabilities to tuple if it exists (stored as numpy array)
        weights = None
        if hasattr(hp, "probabilities") and hp.probabilities is not None:
            weights = tuple(hp.probabilities)
        return Categorical(
            name=new_name,
            items=list(hp.choices),
            default=hp.default_value,
            weights=weights,
            meta=meta,
        )
    elif isinstance(hp, OrdinalHP):
        from ConfigSpace import OrdinalHyperparameter

        return OrdinalHyperparameter(
            name=new_name,
            sequence=list(hp.sequence),
            default=hp.default_value,
            meta=meta,
        )
    elif isinstance(hp, ConstantHP):
        return Constant(
            name=new_name,
            value=hp.value,
            meta=meta,
        )
    else:
        raise TypeError(f"Unknown hyperparameter type: {type(hp)}")


def _clone_condition(condition, prefix: str, hp_map: dict[str, Hyperparameter]):
    """
    Clone a condition with prefixed hyperparameter references.

    Parameters
    ----------
    condition : Condition
        The condition to clone.
    prefix : str
        The prefix that was added to hyperparameter names.
    hp_map : dict
        Mapping from original names to prefixed hyperparameters.
    """
    from ConfigSpace.conditions import (
        EqualsCondition,
        NotEqualsCondition,
        LessThanCondition,
        GreaterThanCondition,
        InCondition,
        AndConjunction,
        OrConjunction,
    )

    if isinstance(condition, AndConjunction):
        return AndConjunction(
            *[_clone_condition(c, prefix, hp_map) for c in condition.components]
        )
    elif isinstance(condition, OrConjunction):
        return OrConjunction(
            *[_clone_condition(c, prefix, hp_map) for c in condition.components]
        )

    # Get prefixed child and parent hyperparameters
    child = hp_map[condition.child.name]
    parent = hp_map[condition.parent.name]

    if isinstance(condition, EqualsCondition):
        return EqualsCondition(child=child, parent=parent, value=condition.value)
    elif isinstance(condition, NotEqualsCondition):
        return NotEqualsCondition(child=child, parent=parent, value=condition.value)
    elif isinstance(condition, LessThanCondition):
        return LessThanCondition(child=child, parent=parent, value=condition.value)
    elif isinstance(condition, GreaterThanCondition):
        return GreaterThanCondition(child=child, parent=parent, value=condition.value)
    elif isinstance(condition, InCondition):
        return InCondition(child=child, parent=parent, values=condition.values)
    else:
        raise TypeError(f"Unknown condition type: {type(condition)}")


def _clone_forbidden(forbidden, prefix: str, hp_map: dict[str, Hyperparameter]):
    """
    Clone a forbidden clause with prefixed hyperparameter references.
    """
    from ConfigSpace.forbidden import (
        ForbiddenEqualsClause,
        ForbiddenInClause,
        ForbiddenAndConjunction,
    )

    if isinstance(forbidden, ForbiddenAndConjunction):
        return ForbiddenAndConjunction(
            *[_clone_forbidden(f, prefix, hp_map) for f in forbidden.components]
        )
    elif isinstance(forbidden, ForbiddenEqualsClause):
        hp = hp_map[forbidden.hyperparameter.name]
        return ForbiddenEqualsClause(hyperparameter=hp, value=forbidden.value)
    elif isinstance(forbidden, ForbiddenInClause):
        hp = hp_map[forbidden.hyperparameter.name]
        return ForbiddenInClause(hyperparameter=hp, values=forbidden.values)
    else:
        raise TypeError(f"Unknown forbidden type: {type(forbidden)}")


class ConfigurableMixin:
    """
    A mixin that provides automatic configuration space generation.

    Classes using this mixin should:
    1. Define a class attribute `PREFIX` (str)
    2. Implement `_define_hyperparameters()` returning (hyperparameters, conditions, forbiddens)

    The mixin provides default implementations of:
    - get_configuration_space(): Builds ConfigSpace with proper prefixes
    - get_from_configuration(): Extracts values and returns a partial

    Example
    -------
    class MySelector(AbstractSelector, ConfigurableMixin):
        PREFIX = "my_selector"

        @staticmethod
        def _define_hyperparameters():
            from ConfigSpace import Categorical, Integer

            model = ClassChoice("model", [ModelA, ModelB])
            num_iter = Integer("num_iter", (10, 100), default=50)

            hyperparameters = [model, num_iter]
            conditions = []
            forbiddens = []

            return hyperparameters, conditions, forbiddens
    """

    PREFIX: str  # Must be defined by subclass

    @staticmethod
    def _define_hyperparameters(
        **kwargs,
    ) -> tuple[
        list[Hyperparameter],
        list,  # conditions
        list,  # forbiddens
    ]:
        """
        Define hyperparameters for this class.

        Override this method to define hyperparameters WITHOUT prefixes.
        The mixin will handle all prefix management.

        Returns
        -------
        tuple
            (hyperparameters, conditions, forbiddens) where:
            - hyperparameters: list of ConfigSpace hyperparameters (including ClassChoice)
            - conditions: list of ConfigSpace conditions (using unprefixed hyperparameters)
            - forbiddens: list of ConfigSpace forbidden clauses
        """
        return [], [], []

    @staticmethod
    def _resolve_class_from_hp(hp: Hyperparameter, value: Any) -> type | bool | None:
        """
        Resolve a class choice from a hyperparameter and its value.

        Handles both ClassChoice and CategoricalHyperparameter (if it was
        converted from ClassChoice and has __class_choices__ in meta).
        """
        if not CONFIGSPACE_AVAILABLE:
            return None

        from asf.utils.configurable import ClassChoice
        from ConfigSpace.hyperparameters import CategoricalHyperparameter

        if isinstance(hp, ClassChoice):
            return hp.get_class(value)

        if isinstance(hp, CategoricalHyperparameter):
            meta = hp.meta if hasattr(hp, "meta") else getattr(hp, "_meta", None)
            if meta and "__class_choices__" in meta:
                choices = meta["__class_choices__"]
                for choice in choices:
                    if choice is False and value == "False":
                        return False
                    if hasattr(choice, "__name__") and choice.__name__ == value:
                        return choice
        return None

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.

        This method is called by get_from_configuration after stripping prefixes
        from the configuration. Subclasses should override this method instead
        of get_from_configuration if they need custom initialization logic.

        Parameters
        ----------
        clean_config : dict
            Dictionary containing the configuration parameters with prefixes removed.
            Only contains parameters defined in _define_hyperparameters.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        partial
            A partial function that will instantiate the class.
        """
        # Default implementation: merge clean_config into kwargs
        init_kwargs = {**clean_config, **kwargs}
        return partial(cls, **init_kwargs)

    @classmethod
    def get_configuration_space(
        cls,
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
        **kwargs,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for this class with proper prefixes.

        Parameters
        ----------
        cs : ConfigurationSpace or None
            Existing configuration space to add to. If None, creates a new one.
        pre_prefix : str
            Prefix from parent configuration spaces.
        parent_param : Hyperparameter or None
            Parent hyperparameter for conditional activation.
        parent_value : str or None
            Value of parent_param that activates this class's parameters.
        **kwargs
            Additional arguments passed to _define_hyperparameters and child spaces.

        Returns
        -------
        ConfigurationSpace
            The configuration space with all hyperparameters properly prefixed.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install with: pip install 'asf[configspace]'"
            )

        if cs is None:
            cs = ConfigurationSpace()

        # Compute prefix
        if pre_prefix:
            prefix = f"{pre_prefix}:{cls.PREFIX}:"
        else:
            prefix = f"{cls.PREFIX}:"

        # Get hyperparameter definitions, passing kwargs (e.g., model_class)
        hyperparameters, conditions, forbiddens = cls._define_hyperparameters(**kwargs)

        if not hyperparameters:
            return cs

        # Clone hyperparameters with prefixes
        hp_map = {}  # original name -> prefixed hyperparameter
        prefixed_hps = []
        class_choices = []  # Track ClassChoice hyperparameters for recursion

        for hp in hyperparameters:
            prefixed_hp = _clone_hyperparameter(hp, prefix)
            hp_map[hp.name] = prefixed_hp
            prefixed_hps.append(prefixed_hp)

            if isinstance(hp, ClassChoice):
                class_choices.append((hp, prefixed_hp))

        # Clone conditions with prefixed references
        prefixed_conditions = [
            _clone_condition(cond, prefix, hp_map) for cond in conditions
        ]

        # Clone forbiddens with prefixed references
        prefixed_forbiddens = [
            _clone_forbidden(forb, prefix, hp_map) for forb in forbiddens
        ]

        # Add parent conditions if this is a nested configuration
        if parent_param is not None and parent_value is not None:
            parent_conditions = [
                EqualsCondition(child=hp, parent=parent_param, value=parent_value)
                for hp in prefixed_hps
            ]
        else:
            parent_conditions = []

        # Add everything to the configuration space
        cs.add(prefixed_hps)
        cs.add(prefixed_conditions)
        cs.add(prefixed_forbiddens)
        cs.add(parent_conditions)

        # Recursively add configuration spaces for ClassChoice hyperparameters
        for original_hp, prefixed_hp in class_choices:
            for choice_cls in original_hp._class_choices:
                if hasattr(choice_cls, "get_configuration_space"):
                    # Use the prefix up to and including the ClassChoice param
                    child_pre_prefix = f"{prefix}{original_hp.name}"
                    choice_cls.get_configuration_space(
                        cs=cs,
                        pre_prefix=child_pre_prefix,
                        parent_param=prefixed_hp,
                        parent_value=choice_cls.__name__,
                        **kwargs,
                    )

        return cs

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict,
        pre_prefix: str = "",
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration to extract values from.
        pre_prefix : str
            Prefix from parent configuration spaces.
        **kwargs
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        partial
            A partial function that will instantiate the class with configured parameters.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install with: pip install 'asf[configspace]'"
            )

        # Compute prefix
        if pre_prefix:
            prefix = f"{pre_prefix}:{cls.PREFIX}:"
        else:
            prefix = f"{cls.PREFIX}:"

        # Get hyperparameter definitions, passing kwargs (e.g., model_class)
        hyperparameters, _, _ = cls._define_hyperparameters(**kwargs)

        # Extract values from configuration
        init_kwargs = {}

        for hp in hyperparameters:
            prefixed_name = f"{prefix}{hp.name}"

            if prefixed_name not in configuration:
                continue

            value = configuration[prefixed_name]

            # Try to resolve class from ClassChoice or converted Categorical
            resolved = cls._resolve_class_from_hp(hp, value)
            if resolved is not None:
                chosen_cls = resolved
                child_pre_prefix = f"{prefix}{hp.name}"

                if hasattr(chosen_cls, "get_from_configuration"):
                    # Get a partial for the chosen class
                    value = chosen_cls.get_from_configuration(
                        configuration=configuration,
                        pre_prefix=child_pre_prefix,
                    )
                else:
                    value = chosen_cls

            init_kwargs[hp.name] = value

        # Call the hook method with clean configuration
        return cls._get_from_clean_configuration(init_kwargs, **kwargs)
