from asf.selectors import PairwiseClassifier
from asf.selectors import SelectorPipeline
from sklearn.ensemble import RandomForestClassifier
from asf.pre_selector import MarginalContributionBasedPreSelector
from asf.preprocessing import get_default_preprocessor
from asf.metrics import virtual_best_solver
from asf.presolving.asap_v2 import ASAPv2
import pandas as pd
import numpy as np


def get_data():
    """
    Generates synthetic data for algorithm scheduling with balanced performance.

    This version of the function fine-tunes the performance curves to prevent any
    single algorithm from dominating. Each algorithm now has a specific, narrow
    set of conditions where it performs optimally, forcing a scheduler to use
    a mix of algorithms to achieve high coverage and low solve times.

    Returns:
        tuple: A tuple containing two pandas DataFrames.
               - The first DataFrame contains the problem features.
               - The second DataFrame contains the performance times for each algorithm.
    """
    np.random.seed(42)
    n_instances = 100

    feature_data = []
    performance_data = []

    for i in range(n_instances):
        complexity = np.random.uniform(1, 10)
        size = np.random.uniform(1, 10)
        density = np.random.uniform(0, 1)
        structure = np.random.uniform(0, 10)
        noise = np.random.normal(0, 0.3)

        features = [complexity, size, density, structure, noise]
        feature_data.append(features)

        # algo1: Good on low complexity
        if complexity < 3:
            algo1_time = 7 + complexity * 2 + size * 1 + np.random.exponential(8)
        else:
            algo1_time = 28 + complexity * 5 + size * 3 + np.random.exponential(20)

        # algo2: Good on high structure
        if structure > 7:
            algo2_time = 5 + complexity * 1.5 + size * 2 + np.random.exponential(10)
        else:
            algo2_time = 25 + complexity * 4 + size * 4 + np.random.exponential(18)

        # algo3: Good on medium density
        if 0.4 < density < 0.7:
            algo3_time = 5 + complexity * 2.5 + size * 1.5 + np.random.exponential(9)
        else:
            algo3_time = 22 + complexity * 4.5 + size * 3.5 + np.random.exponential(19)

        # algo4: Good on high density
        if density > 0.7:
            algo4_time = 6 + complexity * 2 + size * 2 + np.random.exponential(11)
        else:
            algo4_time = 24 + complexity * 4.2 + size * 4 + np.random.exponential(21)

        # algo5: Good on large size
        if size > 7:
            algo5_time = 9 + complexity * 1.8 + size * 1.2 + np.random.exponential(12)
        else:
            algo5_time = 26 + complexity * 3.8 + size * 4.5 + np.random.exponential(22)

        times = [
            max(5, t)
            for t in [algo1_time, algo2_time, algo3_time, algo4_time, algo5_time]
        ]
        performance_data.append(times)

    features = pd.DataFrame(
        feature_data, columns=["complexity", "size", "density", "structure", "noise"]
    )

    performance = pd.DataFrame(
        performance_data, columns=["algo1", "algo2", "algo3", "algo4", "algo5"]
    )

    return features, performance


if __name__ == "__main__":
    # Load the data
    features, performance = get_data()

    # Random train/test split
    np.random.seed(123)
    indices = np.random.permutation(len(features))
    train_size = int(0.7 * len(features))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_features = features.iloc[train_idx]
    train_performance = performance.iloc[train_idx]
    test_features = features.iloc[test_idx]
    test_performance = performance.iloc[test_idx]

    print(
        f"Dataset: {len(features)} instances ({train_size} train, {len(test_features)} test)"
    )

    best_algorithms = train_performance.idxmin(axis=1).value_counts()
    print("Best algorithm distribution (train):", dict(best_algorithms))

    selector = SelectorPipeline(
        selector=PairwiseClassifier(model_class=RandomForestClassifier),
        preprocessor=get_default_preprocessor(),
        algorithm_pre_selector=MarginalContributionBasedPreSelector(
            metric=virtual_best_solver, n_algorithms=3
        ),
        pre_solving=ASAPv2(
            runcount_limit=60.0, budget=50, regularization_weight=0.15, verbosity=1
        ),
    )

    print("\nFitting pipeline...")
    selector.fit(train_features, train_performance)

    predictions = selector.predict(test_features)

    print("\nASAPv2 learned schedule:")
    asap_config = selector.pre_solving.get_preschedule_config()

    # Calculate coverage on test set
    solvable_by_alg = {}
    for alg, time in asap_config.items():
        solvable_by_alg[alg] = set(
            test_performance[test_performance[alg] <= time].index
        )

    all_solvable = set()
    for solvable_set in solvable_by_alg.values():
        all_solvable.update(solvable_set)

    for alg, time in asap_config.items():
        solved = len(solvable_by_alg[alg])
        solve_rate = solved / len(test_performance) * 100

        # Find instances only this algorithm can solve in the schedule
        unique_instances = solvable_by_alg[alg].copy()
        for other_alg in asap_config.keys():
            if other_alg != alg:
                unique_instances -= solvable_by_alg[other_alg]
        unique_count = len(unique_instances)

        print(
            f"  {alg}: {time:.1f}s (solves {solved}/{len(test_performance)} = {solve_rate:.0f}%, unique: {unique_count})"
        )

    total_solved = len(all_solvable)
    schedule_coverage = total_solved / len(test_performance) * 100
    print(
        f"Schedule coverage: {total_solved}/{len(test_performance)} ({schedule_coverage:.0f}%)"
    )

    # Show compact examples
    print("\nExample predictions (test set):")
    preschedule_str = " → ".join(
        [f"{alg}:{time:.0f}s" for alg, time in asap_config.items()]
    )

    for i, (instance_id, prediction) in enumerate(list(predictions.items())[:3]):
        main_alg = (
            prediction[0][0]
            if isinstance(prediction, list) and len(prediction) > 0
            else str(prediction)
        )
        best_actual = test_performance.loc[instance_id].idxmin()
        best_time = test_performance.loc[instance_id].min()

        print(
            f"  {instance_id}: [{preschedule_str}] → {main_alg} (best: {best_actual}:{best_time:.0f}s)"
        )
