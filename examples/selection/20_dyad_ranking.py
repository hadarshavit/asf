import numpy as np
import pandas as pd
from typing import cast

from asf.selectors.dyad_ranking import DyadRanking
from asf.selectors.simple_ranking import SimpleRanking
from asf.metrics import (
    compute_solve_rate,
    single_best_solver,
    virtual_best_solver,
    running_time_selector_performance,
)


def generate_synthetic_data(
    n_instances=400,
    n_algorithms=8,
    n_inst_features=10,
    n_algo_features=6,
    seed=42,
    budget=300.0,
):
    """
    Generate synthetic algorithm selection data where performance depends on
    the interaction between instance features and algorithm features.

    Each algorithm has different "strengths" (algorithm features) and each instance
    has different "characteristics" (instance features). The runtime is computed
    based on how well the algorithm's strengths match the instance's characteristics.

    Parameters
    ----------
    n_instances : int
        Number of problem instances
    n_algorithms : int
        Number of algorithms
    n_inst_features : int
        Number of instance features
    n_algo_features : int
        Number of algorithm features
    seed : int
        Random seed
    budget : float
        Time budget (timeouts will be set to budget * 10)

    Returns
    -------
    features : pd.DataFrame
        Instance features
    performance : pd.DataFrame
        Algorithm runtimes per instance
    algo_features : pd.DataFrame
        Algorithm features
    """
    rng = np.random.RandomState(seed)

    # Generate instance features (problem characteristics)
    # Features represent: complexity, structure, size, etc.
    instance_features = pd.DataFrame(
        rng.uniform(0, 1, size=(n_instances, n_inst_features)),
        columns=[f"inst_feat_{i}" for i in range(n_inst_features)],
        index=[f"instance_{i}" for i in range(n_instances)],
    )

    # Generate algorithm features (solver characteristics)
    # Features represent: heuristic type, parameter values, search strategy, etc.
    algo_features_data = {}
    for i in range(n_algo_features):
        if i < 3:
            # Continuous parameters (e.g., learning rate, temperature)
            algo_features_data[f"param_{i}"] = rng.uniform(0, 1, n_algorithms)
        elif i < 5:
            # Discrete strategy choices (normalized to 0-1)
            algo_features_data[f"strategy_{i - 3}"] = rng.choice(
                [0.0, 0.5, 1.0], n_algorithms
            )
        else:
            # Binary flags (e.g., uses preprocessing, uses caching)
            algo_features_data[f"flag_{i - 5}"] = rng.choice([0.0, 1.0], n_algorithms)

    algo_features = pd.DataFrame(
        algo_features_data,
        index=[f"algo_{i}" for i in range(n_algorithms)],
    )

    # Generate performance based on instance-algorithm interaction
    # Key idea: Each algorithm has affinity for different instance types
    performance = pd.DataFrame(
        index=instance_features.index,
        columns=algo_features.index,
    )

    # Create interaction weights: which instance features matter for which algorithm features
    # This simulates that different algorithms excel at different problem types
    interaction_matrix = rng.randn(n_inst_features, n_algo_features)

    for inst_idx in instance_features.index:
        inst_vec = instance_features.loc[inst_idx].values

        for algo_idx in algo_features.index:
            algo_vec = algo_features.loc[algo_idx].values

            # Compute base runtime as interaction between instance and algorithm
            # Lower dot product = better match = faster runtime
            interaction = np.dot(inst_vec, interaction_matrix @ algo_vec)

            # Add algorithm-specific bias (some algorithms are generally faster/slower)
            algo_bias = 50 + algo_vec.mean() * 100

            # Add instance-specific complexity
            inst_complexity = 20 * inst_vec.mean()

            # Add noise
            noise = rng.normal(0, 10)

            # Compute runtime (ensure positive)
            runtime = max(5.0, algo_bias + inst_complexity + interaction * 50 + noise)

            # Randomly timeout some hard combinations (5-15% depending on algorithm)
            timeout_prob = 0.05 + algo_vec[-1] * 0.1  # Flag_0 controls timeout rate
            if rng.rand() < timeout_prob:
                runtime = budget * 10

            performance.at[inst_idx, algo_idx] = runtime

    return instance_features, performance, algo_features


def main():
    """
    Demonstrate DyadRanking on synthetic data where performance is a function
    of instance and algorithm features.

    This allows the dyad ranking model to learn meaningful relationships between
    problem characteristics and solver parameters.
    """

    print("=" * 70)
    print("Dyad Ranking for Algorithm Selection")
    print("=" * 70)
    print("Based on: Tornede et al. (2019)")
    print("'Algorithm Selection as Recommendation: From Collaborative")
    print(" Filtering to Dyad Ranking'")
    print("=" * 70)
    print()
    print("Using synthetic data where performance depends on the interaction")
    print("between instance features and algorithm features.")
    print()

    # Generate synthetic data
    budget = 300.0
    features, performance, algo_features = generate_synthetic_data(
        n_instances=400,
        n_algorithms=8,
        n_inst_features=10,
        n_algo_features=6,
        seed=42,
        budget=budget,
    )

    print(
        f"Dataset size: {len(features)} instances, {len(performance.columns)} algorithms"
    )
    print(f"Instance features: {features.shape[1]} dimensions")
    print(f"Algorithm features: {algo_features.shape[1]} dimensions")
    print(f"Budget: {budget}s")
    print()
    print("Sample algorithm features:")
    print(algo_features.head(3))
    print()

    # Train/test split
    n_train = int(0.7 * len(features))
    X_train, X_test = features.iloc[:n_train], features.iloc[n_train:]
    Y_train, Y_test = performance.iloc[:n_train], performance.iloc[n_train:]

    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")
    print()

    # Baselines
    sbs_score = single_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    vbs_score = virtual_best_solver(Y_test, maximize=False, budget=budget, par=10.0)
    oracle_sr = float((Y_test.min(axis=1) <= budget).mean())

    sbs_float = cast(float, sbs_score)
    vbs_float = cast(float, vbs_score)

    print(f"Single Best Solver (SBS) PAR10: {sbs_float:.2f}")
    print(f"Virtual Best Solver (VBS) PAR10: {vbs_float:.2f}")
    print(f"Virtual Best Solver solve-rate: {oracle_sr:.2%}")
    print("-" * 70)

    # Test 1: Pairwise sampling with default (10 pairs/instance)
    print("\n1. DyadRanking with pairwise sampling (10 pairs):")
    selector_default = DyadRanking(
        algorithm_features=algo_features,
        budget=budget,
        maximize=False,
    )
    selector_default.fit(X_train, Y_train)
    preds_default = selector_default.predict(X_test)
    assert isinstance(preds_default, dict)
    preds_default = cast(dict[str, list[tuple[str, float] | str]], preds_default)

    sr_default = compute_solve_rate(preds_default, Y_test, budget)
    par10_default = running_time_selector_performance(
        preds_default,
        Y_test,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"  Solve-rate: {sr_default:.2%} | PAR10: {par10_default:.2f}")

    # Test 2: More pairs for comparison
    print("\n2. DyadRanking with more pairs (25 pairs):")
    selector_more = DyadRanking(
        algorithm_features=algo_features,
        n_pairs_per_instance=25,
        random_state=42,
        budget=budget,
        maximize=False,
    )
    selector_more.fit(X_train, Y_train)
    preds_more = selector_more.predict(X_test)
    assert isinstance(preds_more, dict)
    preds_more = cast(dict[str, list[tuple[str, float] | str]], preds_more)

    sr_more = compute_solve_rate(preds_more, Y_test, budget)
    par10_more = running_time_selector_performance(
        preds_more, Y_test, budget=budget, par=10.0, return_per_instance=False
    )
    print(f"  Solve-rate: {sr_more:.2%} | PAR10: {par10_more:.2f}")

    # Test 3: One-hot fallback (not recommended, but for comparison)
    print("\n3. DyadRanking with one-hot fallback (not recommended per paper):")
    selector_onehot = DyadRanking(
        algorithm_features=None,  # Triggers warning
        n_pairs_per_instance=10,
        random_state=42,
        budget=budget,
        maximize=False,
    )
    selector_onehot.fit(X_train, Y_train)
    preds_onehot = selector_onehot.predict(X_test)
    assert isinstance(preds_onehot, dict)
    preds_onehot = cast(dict[str, list[tuple[str, float] | str]], preds_onehot)

    sr_onehot = compute_solve_rate(preds_onehot, Y_test, budget)
    par10_onehot = running_time_selector_performance(
        preds_onehot,
        Y_test,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"  Solve-rate: {sr_onehot:.2%} | PAR10: {par10_onehot:.2f}")

    # Test 4: SimpleRanking (baseline for comparison)
    print("\n4. SimpleRanking (full rankings with one-hot encoding):")
    selector_simple = SimpleRanking(
        budget=budget,
        maximize=False,
    )
    selector_simple.fit(X_train, Y_train)
    preds_simple = selector_simple.predict(X_test)
    assert isinstance(preds_simple, dict)
    preds_simple = cast(dict[str, list[tuple[str, float] | str]], preds_simple)

    sr_simple = compute_solve_rate(preds_simple, Y_test, budget)
    par10_simple = running_time_selector_performance(
        preds_simple,
        Y_test,
        budget=budget,
        par=10.0,
        return_per_instance=False,
    )
    print(f"  Solve-rate: {sr_simple:.2%} | PAR10: {par10_simple:.2f}")

    # Show detailed predictions
    print("\n" + "-" * 70)
    print("Detailed predictions (first 10 instances):")
    print()
    assert isinstance(preds_default, dict)

    for inst in list(X_test.index)[:10]:
        sched = preds_default.get(str(inst), [(None, 0.0)])
        assert isinstance(sched, list)
        if len(sched) > 0:
            algo, _ = sched[0]
        else:
            algo = None

        if algo is not None and algo in Y_test.columns:
            rt = Y_test.at[inst, algo]
            best_algo = Y_test.loc[inst].idxmin()

            solved_str = " ✓" if rt <= budget else " ✗"
            optimal_str = " (optimal)" if algo == best_algo else f" (best: {best_algo})"

            print(
                f"{inst}: selected={algo}, runtime={rt:.1f}s{solved_str}{optimal_str}"
            )
        else:
            print(f"{inst}: selected={algo} (invalid)")

    print("\n" + "=" * 70)
    print("Summary:")
    print(
        f"  10 pairs (default):  PAR10={par10_default:.2f}, solve-rate={sr_default:.2%}"
    )
    print(f"  25 pairs:            PAR10={par10_more:.2f}, solve-rate={sr_more:.2%}")
    print(
        f"  One-hot (fallback):  PAR10={par10_onehot:.2f}, solve-rate={sr_onehot:.2%}"
    )
    print(
        f"  SimpleRanking:       PAR10={par10_simple:.2f}, solve-rate={sr_simple:.2%}"
    )
    print()
    print(f"  Gap to VBS:          {float(par10_default) - vbs_float:.2f}")
    print(
        f"  Improvement over SBS: {((sbs_float - float(par10_default)) / sbs_float * 100):.1f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
