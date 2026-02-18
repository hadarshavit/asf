import pandas as pd
import pytest

from asf.selectors.dyad_ranking import DyadRanking


@pytest.fixture
def algo_features(dummy_performance):
    """Algorithm features fixture for testing."""
    return pd.DataFrame(
        [[1.0, 0.5, 0.2], [0.2, 1.0, 0.8]],
        index=dummy_performance.columns,
        columns=["param1", "param2", "component1"],
    )


@pytest.mark.parametrize(
    "n_pairs,algo_feats",
    [(10, None), (5, "use_fixture"), (10, "use_fixture")],
)
def test_dyad_ranking_basic(
    dummy_performance,
    dummy_features,
    validate_predictions,
    algo_features,
    n_pairs,
    algo_feats,
):
    """Test DyadRanking with various configurations."""
    af = algo_features if algo_feats == "use_fixture" else None
    selector = DyadRanking(
        algorithm_features=af, n_pairs_per_instance=n_pairs, random_state=42
    )
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_dyad_ranking_multiple_algorithms():
    """Test with 3+ algorithms and verify algorithm feature preservation."""
    performance = pd.DataFrame(
        {
            "algo1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "algo2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "algo3": [3.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    algo_features = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        index=["algo1", "algo2", "algo3"],
        columns=["af1", "af2"],
    )

    selector = DyadRanking(algorithm_features=algo_features, n_pairs_per_instance=5)
    selector.fit(features, performance)

    # Verify algorithm features preserved
    assert selector.algorithm_features is not None
    assert list(selector.algorithm_features.index) == ["algo1", "algo2", "algo3"]

    predictions = selector.predict(features)
    assert len(predictions) == len(features)
    for inst_pred in predictions.values():
        assert len(inst_pred) == 1
        assert inst_pred[0][0] in ["algo1", "algo2", "algo3"]


@pytest.mark.parametrize("n_pairs", [25, 5])
def test_dyad_ranking_parameters(n_pairs):
    """Test parameter storage and identical performance handling."""
    selector = DyadRanking(n_pairs_per_instance=n_pairs)
    assert selector.n_pairs_per_instance == n_pairs

    # Test with identical performance on some instances
    performance = pd.DataFrame(
        {
            "algo1": [1.0, 2.0, 3.0, 3.0, 3.0],
            "algo2": [2.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})

    selector.fit(features, performance)
    predictions = selector.predict(features)

    assert len(predictions) == len(features)
    for schedule in predictions.values():
        assert len(schedule) == 1
        algo, budget = schedule[0]
        assert isinstance(algo, str)
        assert isinstance(budget, float)
