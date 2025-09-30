import pandas as pd
import numpy as np
from asf.selectors.collaborative_filtering_selector import (
    CollaborativeFilteringSelector,
)


def generate_correlated_data(n_instances=100, n_algorithms=6, seed=42):
    """
    Generates synthetic performance data for collaborative filtering,
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
    algo_coefs = np.array(
        [
            [10, -10, 20],  # algo1
            [-5, 20, -5],  # algo2
            [15, -5, -100],  # algo3
            [3, 5, 5],  # algo4
            [-10, 8, 100],  # algo5
            [7, -3, 2],  # algo6
        ]
    )

    performance = []
    for i in range(n_instances):
        feats = features.iloc[i].values
        perf_row = algo_coefs @ feats + 200 + np.random.normal(0, 20, n_algorithms)
        performance.append(perf_row)
    performance = pd.DataFrame(
        performance,
        columns=[f"algo{j + 1}" for j in range(n_algorithms)],
        index=features.index,
    )

    return features, performance


def evaluate_predictions(predictions, true_performance):
    correct = 0
    total = 0
    score_diffs = []
    for instance, pred in predictions.items():
        pred_algo, pred_score = pred[0]
        if instance not in true_performance.index:
            continue
        # Find the true best algorithm and its score (ignoring NaNs)
        row = true_performance.loc[instance]
        if row.isnull().all():
            continue
        true_algo = row.idxmin()
        true_score = row.min()
        total += 1
        if pred_algo == true_algo:
            correct += 1
        score_diffs.append(pred_score - true_score)
    accuracy = correct / total if total > 0 else 0
    avg_score_diff = np.mean(score_diffs) if score_diffs else float("nan")
    return accuracy, avg_score_diff


def print_sample_predictions(predictions, true_performance, label, n=10):
    print(f"\nSample predictions for test set ({label}):")
    for instance in list(true_performance.index)[:n]:
        if instance not in predictions:
            continue
        algo, pred_score = predictions[instance][0]
        true_row = true_performance.loc[instance]
        true_best_algo = true_row.idxmin()
        true_best_score = true_row.min()
        real_pred_score = true_row[algo]
        emoji = "✅" if algo == true_best_algo else "❌"
        print(
            f"{instance}: {algo} (predicted: {pred_score:.2f}, real: {real_pred_score:.2f}) "
            f"| True best: {true_best_algo} ({true_best_score:.2f}) {emoji}"
        )
    print("\n\n")


if __name__ == "__main__":
    features, performance_full = generate_correlated_data()

    print("Full performance matrix (no NaNs):")
    print(performance_full.head(10))

    # Split into train/test
    n_train = int(0.7 * len(features))
    train_idx = features.index[:n_train]
    test_idx = features.index[n_train:]

    train_features = features.loc[train_idx]
    train_performance_full = performance_full.loc[train_idx]
    test_features = features.loc[test_idx]
    test_performance_full = performance_full.loc[test_idx]

    # Insert NaNs for training and test sets
    missing_rate = 0.3
    train_performance = train_performance_full.copy()
    test_performance = test_performance_full.copy()
    train_nan_mask = np.random.rand(*train_performance.shape) < missing_rate
    test_nan_mask = np.random.rand(*test_performance.shape) < missing_rate
    train_performance[train_nan_mask] = np.nan
    test_performance[test_nan_mask] = np.nan

    print("\nTraining performance matrix (with NaNs):")
    print(train_performance.head(10))

    selector = CollaborativeFilteringSelector(
        n_components=8, n_iter=500, lr=0.001, reg=0.2
    )
    selector.fit(train_features, train_performance)

    # 1. Predict on training set (no NaNs)
    predictions_train = selector.predict(None, None)
    acc_train, diff_train = evaluate_predictions(
        predictions_train, train_performance_full
    )
    print(
        f"\n[TRAIN] Accuracy: {acc_train:.2%}, Avg. predicted-best minus true-best actual: {diff_train:.2f}"
    )
    print_sample_predictions(
        predictions_train, train_performance_full, label="train (full)", n=10
    )

    # 2. Predict on test set (with sparse performance matrix)
    predictions_test_perf = selector.predict(None, test_performance)
    acc_test_perf, diff_test_perf = evaluate_predictions(
        predictions_test_perf, test_performance_full
    )
    print(
        f"[TEST - Sparse Perf] Accuracy: {acc_test_perf:.2%}, Avg. predicted-best minus true-best actual: {diff_test_perf:.2f}"
    )
    print_sample_predictions(
        predictions_test_perf, test_performance_full, label="test (sparse perf)", n=10
    )

    # 3. Predict on test set (cold start, using features only)
    predictions_test_feat = selector.predict(test_features, None)
    acc_test_feat, diff_test_feat = evaluate_predictions(
        predictions_test_feat, test_performance_full
    )
    print(
        f"[TEST - Cold Start] Accuracy: {acc_test_feat:.2%}, Avg. predicted-best minus true-best actual: {diff_test_feat:.2f}"
    )
    print_sample_predictions(
        predictions_test_feat, test_performance_full, label="test (cold start)", n=10
    )
