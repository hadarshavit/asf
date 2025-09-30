import pandas as pd
import numpy as np
from asf.selectors.collaborative_filtering_selector import CollaborativeFilteringSelector

def generate_correlated_data(n_instances=100, n_algorithms=6, missing_rate=0.3, seed=42):
    """
    Generates synthetic sparse performance data for collaborative filtering,
    with algorithm performances correlated to features and more balanced optima.
    """
    np.random.seed(seed)
    features = pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 10, n_instances),
            "feature2": np.random.uniform(0, 5, n_instances),
            "feature3": np.random.uniform(0, 1, n_instances),
        },
        index=[f"instance_{i}" for i in range(n_instances)],
    )

    # Diverse coefficients and offsets for each algorithm
    algo_coefs = np.array([
        [10, -5, 20],    # algo1
        [-8, 12, 5],     # algo2
        [12, 0, -10],    # algo3
        [5, 5, 5],       # algo4
        [-10, 8, 12],    # algo5
        [7, -3, 0],      # algo6
    ])
    algo_offsets = np.array([100, 120, 100, 110, 135, 120])

    performance = []
    for i in range(n_instances):
        feats = features.iloc[i].values
        # Each algorithm's score: linear + offset + more noise
        perf_row = algo_coefs @ feats + algo_offsets + np.random.normal(0, 20, n_algorithms)
        performance.append(perf_row)
    performance = pd.DataFrame(
        performance,
        columns=[f"algo{j+1}" for j in range(n_algorithms)],
        index=features.index,
    )

    # Randomly set some entries to NaN to simulate missing values
    mask = np.random.rand(n_instances, n_algorithms) < missing_rate
    performance[mask] = np.nan

    return features, performance

if __name__ == "__main__":
    features, performance = generate_correlated_data()

    print("Sparse performance matrix:")
    print(performance.head())

    # Split into train/test
    n_train = int(0.7 * len(features))
    train_idx = features.index[:n_train]
    test_idx = features.index[n_train:]

    train_features = features.loc[train_idx]
    train_performance = performance.loc[train_idx]
    test_features = features.loc[test_idx]
    test_performance = performance.loc[test_idx]

    selector = CollaborativeFilteringSelector(n_components=5, n_iter=300, lr=0.001, reg=0.1)
    selector.fit(train_features, train_performance)

    predictions = selector.predict(None, None)
    print("\nPredicted best algorithms for training set:")
    for instance in train_features.index[:10]:
        algo, score = predictions[instance][0]
        print(f"{instance}: {algo} (predicted score: {score:.2f})")

    predictions = selector.predict(None, test_performance)
    print("\nPredicted best algorithms for each test instance (using sparse performance matrix):")
    for instance in test_features.index:
        algo, score = predictions[instance][0]
        print(f"{instance}: {algo} (predicted score: {score:.2f})")

    predictions = selector.predict(test_features, None)
    print("\nPredicted best algorithms for each test instance (using features):")
    for instance in test_features.index:
        algo, score = predictions[instance][0]
        print(f"{instance}: {algo} (predicted score: {score:.2f})")

    print("\nActual performance for test instances:")
    print(test_performance.head(10))