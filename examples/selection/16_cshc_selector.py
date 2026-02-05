import numpy as np
import pandas as pd

from asf.selectors.cshc import CSHCSelector
from asf.selectors.osl_linear import OSLLinearSelector
from asf.selectors.survival_analysis import SurvivalAnalysis
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def make_data(n_instances=300, n_algorithms=5, n_inst_feats=10, seed=42, budget=500.0):
    """Generates synthetic data for the example."""
    rng = np.random.RandomState(seed)

    features = pd.DataFrame(
        rng.normal(size=(n_instances, n_inst_feats)),
        columns=[f"f{i}" for i in range(n_inst_feats)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    alg_names = [f"algo{i + 1}" for i in range(n_algorithms)]

    perf = pd.DataFrame(index=features.index)
    inst_factors = rng.normal(size=(n_instances, 3))
    alg_factors = rng.normal(size=(n_algorithms, 3))

    for i, algo_name in enumerate(alg_names):
        base_runtime = np.dot(inst_factors, alg_factors[i, :]) * 30 + 150
        noise = rng.normal(0, 20, size=n_instances)
        runtimes = np.clip(base_runtime + noise, 1.0, None)

        runtimes *= 1 - 0.1 * i

        timeout_mask = rng.rand(n_instances) < (0.1 + 0.05 * i)
        runtimes[timeout_mask] = budget * 2
        perf[algo_name] = runtimes

    return features, perf


def main():
    """
    Main function to run the CSHCSelector example with synthetic data.
    """
    budget = 500.0
    X, Y = make_data(n_instances=400, n_algorithms=6, seed=1, budget=budget)

    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    # Define a primary selector that does not require algorithm features.
    primary_selector = OSLLinearSelector(
        budget=budget, reg=1e-3, optimizer_method="L-BFGS-B", maxiter=500
    )

    # Define a backup selector (optional, can be None).
    backup_selector = SurvivalAnalysis(budget=budget)

    # Initialize the CSHC selector, providing the budget at initialization
    sel = CSHCSelector(
        primary_selector=primary_selector,
        backup_selector=backup_selector,
        n_folds=3,
        random_state=42,
        budget=budget,
    )

    sel.fit(X_train, Y_train)

    preds = sel.predict(X_test)
    sr = compute_solve_rate(preds, Y_test, budget)

    # Use ASF metrics for baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    par10 = running_time_selector_performance(
        preds, Y_test, budget=budget, par=10.0, return_per_instance=False
    )

    # --- Detailed Evaluation of CSHC Decisions ---
    primary_used = 0
    primary_success = 0
    backup_used = 0
    backup_success = 0
    guardian_override_used = 0
    guardian_override_success = 0

    for inst_name in X_test.index:
        inst_feature_df = X_test.loc[[inst_name]]

        # 1. Get primary selector's choice
        primary_pred = sel.primary_selector.predict(inst_feature_df).get(inst_name)  # type: ignore[attr-defined]
        if not primary_pred:
            continue

        chosen_algo_primary = primary_pred[0][0]

        # 2. Get guardian's confidence in the primary choice
        guardian_for_choice = sel.guardians.get(chosen_algo_primary)
        prob_success = (
            guardian_for_choice.predict_proba(inst_feature_df)[0, 1]
            if guardian_for_choice
            else 0.0
        )

        final_algo = None
        # 3. Trace the decision logic
        if prob_success >= sel.threshold:
            primary_used += 1
            final_algo = chosen_algo_primary
            if Y_test.at[inst_name, final_algo] <= budget:
                primary_success += 1
        elif sel.backup_selector:
            backup_used += 1
            backup_pred_list = sel.backup_selector.predict(inst_feature_df).get(  # type: ignore[attr-defined]
                inst_name
            )
            if backup_pred_list:
                final_algo = backup_pred_list[0][0]
                if Y_test.at[inst_name, final_algo] <= budget:
                    backup_success += 1
        else:  # Guardian override
            guardian_override_used += 1
            best_algo_override = ""
            max_prob = -1.0
            for algo, guardian in sel.guardians.items():
                prob = guardian.predict_proba(inst_feature_df)[0, 1]
                if prob > max_prob:
                    max_prob = prob
                    best_algo_override = algo
            final_algo = best_algo_override
            if Y_test.at[inst_name, final_algo] <= budget:
                guardian_override_success += 1

    # --- Baselines for comparison ---
    # Single best solver on the training set
    best_single_solver = ((Y_train <= budget).mean(axis=0)).idxmax()
    baseline_preds = {idx: [(best_single_solver, budget)] for idx in X_test.index}
    base_sr = compute_solve_rate(baseline_preds, Y_test, budget)

    # Oracle (perfect selector)
    oracle_sr = float((Y_test.min(axis=1) <= budget).mean())

    # --- Print Results ---
    print("=" * 60)
    print("CSHCSelector (synthetic data)")
    print("=" * 60)
    print(f"Train / Test: {len(X_train)} / {len(X_test)} | Budget: {budget}")
    print(f"Primary Selector: {primary_selector.__class__.__name__}")
    print(
        f"Backup Selector: {backup_selector.__class__.__name__ if backup_selector else 'None'}"
    )
    print(f"Learned Confidence Threshold: {sel.threshold:.4f}")
    print()
    print(f"Single Best Solver PAR10 Score: {sbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) PAR10 Score: {vbs_score:.2f}")
    print(f"Virtual Best Solver (Oracle) solve-rate: {oracle_sr:.2%}")
    print()
    print(f"CSHC selector solve-rate: {sr:.2%}")
    print(f"CSHC selector PAR10 Score: {par10:.2f}")
    print(f"Best-single ({best_single_solver}) solve-rate: {base_sr:.2%}")
    print()
    print("--- CSHC Decision Breakdown ---")
    if primary_used > 0:
        print(
            f"Primary selector trusted: {primary_used} times ({primary_success / primary_used:.2%} success)"
        )
    if backup_used > 0:
        print(
            f"Backup selector used:     {backup_used} times ({backup_success / backup_used:.2%} success)"
        )
    if guardian_override_used > 0:
        print(
            f"Guardian override used:   {guardian_override_used} times ({guardian_override_success / guardian_override_used:.2%} success)"
        )
    print()
    print("Sample decisions (first 12):")
    for inst in list(X_test.index)[:12]:
        algo, _ = preds.get(inst, [(None, None)])[0]
        rt = Y_test.at[inst, algo] if algo is not None else float("nan")
        print(f"{inst}: chosen={algo}, true_rt={rt:.2f}")
    print("=" * 60)

    return sel, X_test, Y_test, preds


if __name__ == "__main__":
    main()
