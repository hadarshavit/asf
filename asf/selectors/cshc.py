from __future__ import annotations

import copy
from functools import partial
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class CSHCSelector(ConfigurableMixin, AbstractSelector):
    """
    Confidence-Switching Hybrid Selector.

    A meta-selector that uses a primary selector along with guardian models to
    predict the success probability of the primary's choice.

    Attributes
    ----------
    primary_selector : AbstractSelector
        The primary selector model.
    backup_selector : AbstractSelector or None
        The backup selector model.
    n_folds : int
        Number of folds for cross-validation to find the optimal threshold.
    random_state : int
        Random seed for reproducibility.
    guardian_kwargs : dict[str, Any]
        Keyword arguments for the guardian models (RandomForestClassifier).
    threshold_grid : np.ndarray
        Grid of thresholds to evaluate.
    guardians : dict[str, RandomForestClassifier]
        Trained guardian models for each algorithm.
    threshold : float
        The learned optimal threshold for switching.
    """

    PREFIX = "cshc"
    RETURN_TYPE = "single"

    def __init__(
        self,
        primary_selector: AbstractSelector | Callable[[], AbstractSelector],
        backup_selector: AbstractSelector
        | Callable[[], AbstractSelector]
        | None = None,
        n_estimators: int = 100,
        guardian_kwargs: dict[str, Any] | None = None,
        n_folds: int = 5,
        threshold_grid: np.ndarray | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CSHCSelector.

        Parameters
        ----------
        primary_selector : AbstractSelector or Callable
            The primary selector model.
        backup_selector : AbstractSelector or Callable or None, default=None
            The backup selector model.
        n_estimators : int, default=100
            Number of estimators for the guardian models.
        guardian_kwargs : dict or None, default=None
            Additional keyword arguments for guardian models.
        n_folds : int, default=5
            Number of folds for cross-validation.
        threshold_grid : np.ndarray or None, default=None
            Grid of thresholds to evaluate.
        random_state : int, default=42
            Random seed.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)

        if callable(primary_selector):
            self.primary_selector = primary_selector()
        else:
            self.primary_selector = primary_selector

        if callable(backup_selector):
            self.backup_selector = backup_selector()
        else:
            self.backup_selector = backup_selector

        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.guardian_kwargs = dict(guardian_kwargs or {})
        self.guardian_kwargs.setdefault("n_estimators", int(n_estimators))
        self.guardian_kwargs.setdefault("random_state", int(random_state))
        self.threshold_grid = (
            threshold_grid
            if threshold_grid is not None
            else np.linspace(0.01, 0.99, 99)
        )

        self.guardians: dict[str, RandomForestClassifier] = {}
        self.threshold: float = 0.5

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Train one guardian model per algorithm and find the optimal threshold.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.algorithms = [str(a) for a in performance.columns]
        n_instances = len(features)
        kf = KFold(
            n_splits=min(self.n_folds, n_instances),
            shuffle=True,
            random_state=self.random_state,
        )

        oof_probs = []
        oof_true_success = []

        for train_idx, val_idx in kf.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            Y_train, Y_val = performance.iloc[train_idx], performance.iloc[val_idx]

            sel_copy = copy.deepcopy(self.primary_selector)
            sel_copy.fit(X_train, Y_train)
            primary_preds = sel_copy.predict(X_val)

            fold_guardians = {}
            for algo in self.algorithms:
                y_algo_train = (Y_train[algo] <= self.budget).astype(int)
                clf = RandomForestClassifier(**self.guardian_kwargs)
                clf.fit(X_train, y_algo_train)
                fold_guardians[algo] = clf

            assert isinstance(primary_preds, dict)
            for inst_name, pred_list in primary_preds.items():
                if pred_list:
                    chosen_algo = pred_list[0][0]
                    if Y_val.index.dtype != object:
                        # Convert back to original index type if necessary
                        orig_inst_name = Y_val.index.dtype.type(inst_name)
                    else:
                        orig_inst_name = inst_name

                    runtime = Y_val.at[orig_inst_name, chosen_algo]
                    inst_feature_df = X_val.loc[[orig_inst_name]]
                    guardian_for_choice = fold_guardians.get(chosen_algo)

                    if guardian_for_choice:
                        if len(guardian_for_choice.classes_) == 1:
                            class_val = guardian_for_choice.classes_[0]
                            prob = 1.0 if class_val == 1 else 0.0
                        else:
                            prob = guardian_for_choice.predict_proba(inst_feature_df)[
                                0, 1
                            ]

                        oof_probs.append(float(prob))
                        oof_true_success.append(
                            1 if pd.notna(runtime) and runtime <= self.budget else 0
                        )

        oof_probs_arr = np.array(oof_probs)
        oof_true_success_arr = np.array(oof_true_success)

        best_t, best_ratio = 0.5, float("inf")
        for t in self.threshold_grid:
            preds = (oof_probs_arr >= t).astype(int)
            fn = ((oof_true_success_arr == 1) & (preds == 0)).sum()
            tn = ((oof_true_success_arr == 0) & (preds == 0)).sum()
            ratio = fn / tn if tn > 0 else float("inf")
            if ratio < best_ratio:
                best_ratio, best_t = float(ratio), float(t)
        self.threshold = best_t

        for algo in self.algorithms:
            y_algo = (performance[algo] <= self.budget).astype(int)
            self.guardians[str(algo)] = RandomForestClassifier(
                **self.guardian_kwargs
            ).fit(features, y_algo)

        self.primary_selector.fit(features, performance)
        if self.backup_selector:
            self.backup_selector.fit(features, performance)

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict using the guardian/backup logic.

        Parameters
        ----------
        features : pd.DataFrame or None
            The query instance features.
        performance : pd.DataFrame or None, default=None
            The performance data.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if features is None:
            raise ValueError("CSHCSelector requires features for prediction.")
        if not self.guardians:
            raise RuntimeError("The selector has not been fitted yet.")

        primary_preds = cast(dict[str, Any], self.primary_selector.predict(features))
        final_preds: dict[str, list[tuple[str, float]]] = {}

        for inst_name in features.index:
            preds = primary_preds.get(str(inst_name), [])
            if not isinstance(preds, list):
                preds = []

            pred_list: list[tuple[str, float]] = []
            for p in preds:
                if isinstance(p, tuple):
                    pred_list.append((str(p[0]), float(p[1])))
                else:
                    pred_list.append((str(p), float(self.budget or 0)))

            inst_feature_df = features.loc[[inst_name]]

            if not pred_list:
                if self.backup_selector:
                    backup_pred = cast(
                        dict[str, Any], self.backup_selector.predict(inst_feature_df)
                    )
                    preds = backup_pred.get(str(inst_name), [])
                    if preds:
                        final_preds[str(inst_name)] = [
                            (str(preds[0][0]), float(self.budget or 0))
                        ]
                    else:
                        final_preds[str(inst_name)] = []
                else:
                    final_preds[str(inst_name)] = []
                continue

            chosen_algo_primary = pred_list[0][0]

            guardian_for_choice = self.guardians.get(str(chosen_algo_primary))
            prob_success = 0.0
            if guardian_for_choice:
                if len(guardian_for_choice.classes_) == 1:
                    prob_success = 1.0 if guardian_for_choice.classes_[0] == 1 else 0.0
                else:
                    prob_success = float(
                        guardian_for_choice.predict_proba(inst_feature_df)[0, 1]
                    )

            if prob_success >= self.threshold:
                final_preds[str(inst_name)] = [
                    (str(chosen_algo_primary), float(self.budget or 0))
                ]
            elif self.backup_selector:
                backup_pred = cast(
                    dict[str, Any], self.backup_selector.predict(inst_feature_df)
                )
                preds = backup_pred.get(str(inst_name), [])
                if preds:
                    final_preds[str(inst_name)] = [
                        (str(preds[0][0]), float(self.budget or 0))
                    ]
                else:
                    final_preds[str(inst_name)] = []
            else:
                best_algo = chosen_algo_primary
                max_prob = -1.0
                for algo, guardian in self.guardians.items():
                    if len(guardian.classes_) == 1:
                        prob = 1.0 if guardian.classes_[0] == 1 else 0.0
                    else:
                        prob = float(guardian.predict_proba(inst_feature_df)[0, 1])
                    if prob > max_prob:
                        max_prob = prob
                        best_algo = algo
                final_preds[str(inst_name)] = [
                    (str(best_algo), float(self.budget or 0))
                ]

        return final_preds

    @staticmethod
    def _define_hyperparameters(
        candidate_selectors: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for CSHCSelector.

        Parameters
        ----------
        candidate_selectors : list[type] or None, default=None
            List of selector classes to choose from.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if candidate_selectors is None:
            return [], [], []

        primary_selector_param = ClassChoice(
            name="primary_selector",
            choices=candidate_selectors,
            default=candidate_selectors[0],
        )

        use_backup = Categorical(
            name="use_backup_selector",
            items=[True, False],
            default=False,
        )

        backup_selector_param = ClassChoice(
            name="backup_selector",
            choices=candidate_selectors,
            default=candidate_selectors[0],
        )

        n_estimators_param = Integer(
            name="n_estimators",
            bounds=(10, 200),
            default=100,
        )

        n_folds_param = Integer(
            name="n_folds",
            bounds=(2, 10),
            default=5,
        )

        params = [
            primary_selector_param,
            use_backup,
            backup_selector_param,
            n_estimators_param,
            n_folds_param,
        ]

        conditions = [
            EqualsCondition(backup_selector_param, use_backup, True),  # type: ignore[arg-type]
        ]

        return params, conditions, []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[CSHCSelector]:
        """
        Create a partial function from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            The clean configuration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for CSHCSelector.
        """
        config = clean_config.copy()

        if not config.get("use_backup_selector", False):
            config["backup_selector"] = None

        if "use_backup_selector" in config:
            del config["use_backup_selector"]

        config.update(kwargs)
        return partial(CSHCSelector, **config)
