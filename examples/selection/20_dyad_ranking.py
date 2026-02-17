import numpy as np
import pandas as pd
from typing import Sequence, cast

from asf.selectors.dyad_ranking import DyadRanking
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def make_data(n_instances=300, n_algorithms=6, seed=2, budget=200.0):
    """Generate synthetic features and per-algorithm runtime matrix.

    Runtimes are positive with some timeouts (> budget).
    """
    rng = np.random.RandomState(seed)

    # Instance features
    n_feats = 6
    features = pd.DataFrame(
        rng.normal(size=(n_instances, n_feats)),
        columns=[f"f{i}" for i in range(n_feats)],
        index=[f"inst_{i}" for i in range(n_instances)],
    )

    # Algorithm runtimes derived from linear relation + noise + occasional timeouts
    perf = pd.DataFrame(index=features.index)
    for a in range(n_algorithms):
        bias = rng.uniform(20, 120) * (1 + 0.25 * (a % 2))
        coeff = rng.uniform(-3, 3, size=n_feats)
        noise = rng.normal(0, 10, size=n_instances)
        runtimes = np.clip(features.values @ coeff + bias + noise, 1.0, None)
        timeout_mask = rng.rand(n_instances) < (0.10 + 0.03 * (a % 3))
        runtimes[timeout_mask] = budget * 10
        perf[f"algo{a + 1}"] = runtimes

    return features, perf


def create_algorithm_features(algorithm_names, seed=42):
    """Create synthetic algorithm features.
    
    In practice, these could be:
    - Algorithm hyperparameters
    - Heuristic characteristics
    - Historical performance statistics
    """
    rng = np.random.RandomState(seed)
    n_algos = len(algorithm_names)
    
    # Create diverse algorithm characteristics
    algo_feat_data = {
        "complexity": rng.uniform(1, 10, n_algos),
        "memory_usage": rng.uniform(0.1, 5.0, n_algos),
        "randomized": rng.choice([0, 1], n_algos),
        "heuristic_type": rng.uniform(0, 1, n_algos),
    }
    
    algo_features = pd.DataFrame(
        algo_feat_data,
        index=algorithm_names,
    )
    return algo_features


def main():
    budget = 200.0
    X, Y = make_data(n_instances=300, n_algorithms=8, seed=2, budget=budget)

    # Train/test split
    n_train = int(0.7 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]

    print("=" * 70)
    print("Dyad Ranking for Algorithm Selection")
    print("=" * 70)
    print("Based on: Tornede et al. (2019)")
    print("'Algorithm Selection as Recommendation: From Collaborative")
    print(" Filtering to Dyad Ranking'")
    print("=" * 70)
    print(f"Budget: {budget}s")
    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print(f"Algorithms: {list(Y.columns)}")
    print()

    # Baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)

    print(f"Single Best Solver (SBS) PAR10: {sbs_score:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_score:.2f}")
    print("-" * 70)

    # Test 1: Default one-hot algorithm features
    print("\n1. DyadRanking with default one-hot algorithm features:")
    selector_onehot = DyadRanking(
        budget=budget,
        maximize=False,
    )
    selector_onehot.fit(X_train, Y_train)
    preds_onehot = selector_onehot.predict(X_test)
    
    # Convert to budgeted predictions
    budgeted_preds_onehot = {
        inst: [(algo, budget) for algo, _ in sched] 
        for inst, sched in preds_onehot.items()
    }
    
    sr_onehot = compute_solve_rate(budgeted_preds_onehot, Y_test, budget)
    par10_onehot = running_time_selector_performance(
        budgeted_preds_onehot, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_onehot:.2%} | PAR10: {par10_onehot:.2f}")

    # Test 2: Custom algorithm features
    print("\n2. DyadRanking with custom algorithm features:")
    algo_features = create_algorithm_features(Y_train.columns, seed=42)
    print(f"\nAlgorithm features shape: {algo_features.shape}")
    print("Algorithm feature preview:")
    print(algo_features.head())
    print()
    
    selector_custom = DyadRanking(
        algorithm_features=algo_features,
        budget=budget,
        maximize=False,
    )
    selector_custom.fit(X_train, Y_train)
    preds_custom = selector_custom.predict(X_test)
    
    budgeted_preds_custom = {
        inst: [(algo, budget) for algo, _ in sched] 
        for inst, sched in preds_custom.items()
    }
    
    sr_custom = compute_solve_rate(budgeted_preds_custom, Y_test, budget)
    par10_custom = running_time_selector_performance(
        budgeted_preds_custom, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_custom:.2%} | PAR10: {par10_custom:.2f}")

    # Show detailed predictions
    print("\n" + "-" * 70)
    print("Detailed predictions (first 10 instances, custom features):")
    print()
    
    for inst in list(X_test.index)[:10]:
        sched = preds_custom.get(inst, [(None, 0.0)])
        algo, _ = sched[0]
        
        if algo is not None and algo in Y_test.columns:
            rt = Y_test.at[inst, algo]
            best_algo = Y_test.loc[inst].idxmin()
            best_rt = Y_test.loc[inst].min()
            
            solved_str = " ✓" if rt <= budget else " ✗"
            optimal_str = " (optimal)" if algo == best_algo else f" (best: {best_algo})"
            
            print(f"{inst}: selected={algo}, runtime={rt:.1f}s{solved_str}{optimal_str}")
            print(f"  └─ Best possible: {best_algo} at {best_rt:.1f}s")
        else:
            print(f"{inst}: selected={algo} (invalid)")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  One-hot features:  PAR10={par10_onehot:.2f}, solve-rate={sr_onehot:.2%}")
    print(f"  Custom features:   PAR10={par10_custom:.2f}, solve-rate={sr_custom:.2%}")
    print(f"  Gap to VBS:        {par10_custom - vbs_score:.2f}")
    print("=" * 70)
    
    # Explain dyad concept
    print("\nHow Dyad Ranking works:")
    print("  1. Creates 'dyads' = (instance, algorithm) pairs")
    print(f"     → {len(X_train)} instances × {len(Y_train.columns)} algorithms")
    print(f"     = {len(X_train) * len(Y_train.columns)} training dyads")
    print("  2. Each dyad combines:")
    print("     - Instance features (problem characteristics)")
    print("     - Algorithm features (solver characteristics)")
    print("  3. Trains ranking model to predict best algorithms per instance")
    print("  4. At test time: ranks all algorithms for each new instance")
    print("=" * 70)


if __name__ == "__main__":
    main()
