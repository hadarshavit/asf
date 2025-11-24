import copy
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from asf.selectors.abstract_selector import AbstractSelector


class CSHCSelector(AbstractSelector):
    """
    CSHC: Confidence-Switching Hybrid Selector.

    A meta-selector that uses a primary selector along with guardian models to
    predict the success probability of the primary's choice. If the confidence
    is below a learned threshold, it can fall back to a backup selector or
    choose the algorithm with the highest predicted success probability.
    """

    PREFIX = "cshc"
    RETURN_TYPE = "single"

    def __init__(
        self,
        primary_selector: AbstractSelector,
        backup_selector: Optional[AbstractSelector] = None,
        n_estimators: int = 100,
        guardian_kwargs: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        threshold_grid: Optional[np.ndarray] = None,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if getattr(primary_selector, "RETURN_TYPE", None) != "single":
            raise ValueError("Primary selector must have RETURN_TYPE 'single'.")
        if (
            backup_selector
            and getattr(backup_selector, "RETURN_TYPE", None) != "single"
        ):
            raise ValueError("Backup selector must have RETURN_TYPE 'single'.")

        self.primary_selector = primary_selector
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

        self.guardians: Dict[str, RandomForestClassifier] = {}
        self.threshold: float = 0.5
        self.algorithms: List[str] = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs) -> None:
        """
        Train one guardian model per algorithm and find the optimal threshold.
        """
        self.algorithms = list(performance.columns)
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

            # OOF Choice Generator
            sel_copy = copy.deepcopy(self.primary_selector)
            sel_copy.fit(X_train, Y_train)
            primary_preds = sel_copy.predict(X_val)

            # OOF Probability Generator
            fold_guardians = {}
            for algo in self.algorithms:
                y_algo_train = (Y_train[algo] <= self.budget).astype(int)
                clf = RandomForestClassifier(**self.guardian_kwargs)
                clf.fit(X_train, y_algo_train)
                fold_guardians[algo] = clf

            # Calculate OOF Probabilities for the Validation Set
            for inst_name, pred_list in primary_preds.items():
                if pred_list:
                    chosen_algo = pred_list[0][0]
                    runtime = Y_val.at[inst_name, chosen_algo]

                    inst_feature_df = X_val.loc[[inst_name]]

                    guardian_for_choice = fold_guardians.get(chosen_algo)

                    if guardian_for_choice:
                        prob = guardian_for_choice.predict_proba(inst_feature_df)[0, 1]
                        oof_probs.append(prob)
                        oof_true_success.append(
                            1 if pd.notna(runtime) and runtime <= self.budget else 0
                        )

        oof_probs = np.array(oof_probs)
        oof_true_success = np.array(oof_true_success)

        # Mimize false negatives / true negatives ratio
        best_t, best_ratio = 0.5, float("inf")
        for t in self.threshold_grid:
            preds = (oof_probs >= t).astype(int)
            fn = ((oof_true_success == 1) & (preds == 0)).sum()
            tn = ((oof_true_success == 0) & (preds == 0)).sum()
            ratio = fn / tn if tn > 0 else float("inf")
            if ratio < best_ratio:
                best_ratio, best_t = ratio, t
        self.threshold = best_t

        # Train one guardian for each algorithm on the full dataset
        for algo in self.algorithms:
            y_algo = (performance[algo] <= self.budget).astype(int)
            self.guardians[algo] = RandomForestClassifier(**self.guardian_kwargs).fit(
                features, y_algo
            )

        self.primary_selector.fit(features, performance)
        if self.backup_selector:
            self.backup_selector.fit(features, performance)

    def _predict(self, features: pd.DataFrame) -> Dict[str, List[tuple[str, float]]]:
        """
        Predict using the guardian/backup logic.
        """
        if not self.guardians:
            raise RuntimeError("The selector has not been fitted yet.")

        primary_preds = self.primary_selector.predict(features)
        final_preds = {}

        for inst_name, pred_list in primary_preds.items():
            inst_feature_df = features.loc[[inst_name]]

            if not pred_list:
                if self.backup_selector:
                    backup_pred = self.backup_selector.predict(inst_feature_df)
                    chosen_algo = backup_pred.get(inst_name, [(None, 0)])[0][0]
                    final_preds[inst_name] = [(chosen_algo, self.budget)]
                else:
                    final_preds[inst_name] = []
                continue

            chosen_algo_primary = pred_list[0][0]

            guardian_for_choice = self.guardians.get(chosen_algo_primary)
            prob_success = (
                guardian_for_choice.predict_proba(inst_feature_df)[0, 1]
                if guardian_for_choice
                else 0.0
            )

            if prob_success >= self.threshold:
                final_preds[inst_name] = [(chosen_algo_primary, self.budget)]
            elif self.backup_selector:
                backup_pred = self.backup_selector.predict(inst_feature_df)
                chosen_algo = backup_pred.get(inst_name, [(None, 0)])[0][0]
                final_preds[inst_name] = [(chosen_algo, self.budget)]
            else:
                # No backup: query all guardians and pick solver with highest success probability
                best_algo = chosen_algo_primary
                max_prob = -1.0
                for algo, guardian in self.guardians.items():
                    prob = guardian.predict_proba(inst_feature_df)[0, 1]
                    if prob > max_prob:
                        max_prob = prob
                        best_algo = algo
                final_preds[inst_name] = [(best_algo, self.budget)]

        return final_preds
