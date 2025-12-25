import copy
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Any


class MetaSelector(ConfigurableMixin, AbstractSelector):
    """
    A meta-selector that trains multiple base selectors and uses another selector
    to choose among them for each instance.
    """

    PREFIX = "meta"
    RETURN_TYPE = "single"

    def __init__(
        self,
        base_selectors: List[AbstractSelector],
        meta_selector: AbstractSelector,
        par_factor: int = 10,
        n_folds: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the MetaSelector.

        Args:
            base_selectors (List[AbstractSelector]): A list of algorithm selectors
            meta_selector (AbstractSelector): The selector instance that will be
                trained to choose the best base selector.
            par_factor (int): The factor by which the penalty is increased.
            n_folds (int): The number of folds for cross-validation.
            random_state (int): The random state for reproducibility.
            **kwargs: Additional arguments for the parent class.
        """
        super().__init__(**kwargs)
        if not base_selectors:
            raise ValueError("`base_selectors` list cannot be empty.")
        if not meta_selector:
            raise ValueError("`meta_selector` cannot be None.")

        for sel in base_selectors:
            if getattr(sel, "RETURN_TYPE", None) != "single":
                raise ValueError(
                    f"Base selector {sel.__class__.__name__} must have RETURN_TYPE 'single'."
                )
        if getattr(meta_selector, "RETURN_TYPE", None) != "single":
            raise ValueError("Meta selector must have RETURN_TYPE 'single'.")

        self.base_selectors = base_selectors
        self.meta_selector = meta_selector
        self.par_factor = int(par_factor)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)

        self.selector_names = [
            f"{s.__class__.__name__}_{i}" for i, s in enumerate(self.base_selectors)
        ]
        self._selector_map = {
            name: sel for name, sel in zip(self.selector_names, self.base_selectors)
        }
        self.base_selectors_ = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the MetaSelector using out-of-fold predictions from base selectors.

        Args:
            features (pd.DataFrame): Training features (instances x features).
            performance (pd.DataFrame): Training performance (instances x algorithms).
        """
        penalty = float(self.budget) * float(self.par_factor)
        n_instances = len(features)
        meta_performance = pd.DataFrame(
            index=features.index, columns=self.selector_names, dtype=float
        )
        meta_performance[:] = np.nan

        n_splits = min(self.n_folds, max(2, n_instances))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(np.arange(n_instances)):
            train_ix = features.index[train_idx]
            val_ix = features.index[val_idx]

            X_train = features.loc[train_ix]
            Y_train = performance.loc[train_ix]
            X_val = features.loc[val_ix]

            for sel_idx, sel in enumerate(self.base_selectors):
                try:
                    sel_copy = sel.__class__()
                except Exception:
                    sel_copy = copy.deepcopy(sel)
                sel_copy.fit(X_train, Y_train)
                preds = sel_copy.predict(X_val)

                col = self.selector_names[sel_idx]
                for inst in val_ix:
                    pred = preds.get(inst, [])
                    if not pred:
                        meta_performance.at[inst, col] = penalty
                        continue
                    algo_name = pred[0][0]
                    rt = performance.at[inst, algo_name]
                    if pd.isna(rt) or rt >= self.budget:
                        meta_performance.at[inst, col] = penalty
                    else:
                        meta_performance.at[inst, col] = float(rt)

        meta_performance.fillna(penalty, inplace=True)

        self.base_selectors_ = []
        for sel in self.base_selectors:
            try:
                sel_full = sel.__class__()
            except Exception:
                sel_full = copy.deepcopy(sel)
            sel_full.fit(features, performance)
            self.base_selectors_.append(sel_full)

        self._selector_map = {
            name: sel for name, sel in zip(self.selector_names, self.base_selectors_)
        }

        self.meta_selector.fit(features, meta_performance)

    def _predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        """
        Make predictions with the fitted MetaSelector.

        Args:
            features (pd.DataFrame): Feature matrix for the test instances.

        Returns:
            Dict[str, List[Tuple[str, float]]]: A prediction for each instance.
        """
        meta_predictions = self.meta_selector.predict(features)

        final_predictions = {}
        for instance_name, chosen_selector_list in meta_predictions.items():
            if not chosen_selector_list:
                final_predictions[instance_name] = []
                continue

            chosen_selector_name = chosen_selector_list[0][0]
            chosen_selector = self._selector_map.get(chosen_selector_name)
            if chosen_selector is None:
                final_predictions[instance_name] = []
                continue

            instance_features = features.loc[[instance_name]]
            final_prediction = chosen_selector.predict(instance_features)
            preds = final_prediction.get(instance_name, [])

            formatted = []
            for entry in preds:
                algo, _ = entry
                formatted.append((algo, self.budget))

            final_predictions[instance_name] = formatted
        return final_predictions

    @staticmethod
    def _define_hyperparameters(candidate_selectors=None, **kwargs):
        """Define hyperparameters for MetaSelector."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if candidate_selectors is None:
            return [], [], []

        meta_selector_param = ClassChoice(
            name="meta_selector",
            choices=candidate_selectors,
            default=candidate_selectors[0],
        )

        par_factor_param = Integer(
            name="par_factor",
            bounds=(1, 100),
            default=10,
        )

        n_folds_param = Integer(
            name="n_folds",
            bounds=(2, 10),
            default=5,
        )

        params = [
            meta_selector_param,
            par_factor_param,
            n_folds_param,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        candidate_selectors: List[type] | None = None,
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()

        # Instantiate base selectors with default configuration
        if candidate_selectors:
            base_selectors = []
            for sel_cls in candidate_selectors:
                try:
                    base_selectors.append(sel_cls())
                except Exception:
                    # Fallback if init requires args (should not happen for ConfigurableMixin classes)
                    # We can't do much here if we don't know the args.
                    pass
            config["base_selectors"] = base_selectors

        config.update(kwargs)
        return partial(MetaSelector, **config)
