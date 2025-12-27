from asf.selectors import CosineSelector
import pandas as pd


def test_cosine_selector_default(
    dummy_performance, dummy_features, validate_predictions
):
    selector = CosineSelector()
    # CosineSelector needs algorithm_features passed to fit()
    algo_features = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=dummy_performance.columns,
        columns=["af1", "af2"],
    )
    selector.fit(dummy_features, dummy_performance, algorithm_features=algo_features)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_cosine_selector_custom_params(
    dummy_performance, dummy_features, validate_predictions
):
    selector = CosineSelector(embed_size=32, num_hiddens=32, lr=0.01, num_epochs=2)
    algo_features = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=dummy_performance.columns,
        columns=["af1", "af2"],
    )
    selector.fit(dummy_features, dummy_performance, algorithm_features=algo_features)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)

    assert selector.embed_size == 32
    assert selector.num_hiddens == 32
