import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_performance():
    data = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [1.2, 4.8],
            [2.2, 3.8],
            [3.2, 2.8],
            [4.2, 1.8],
            [5.2, 0.8],
            [0.8, 5.2],
            [1.8, 4.2],
            [2.8, 3.2],
            [3.8, 2.2],
            [4.8, 1.2],
            [1.5, 4.5],
            [2.5, 3.5],
            [3.5, 2.5],
            [4.5, 1.5],
            [5.5, 0.5],
        ]
    )
    return pd.DataFrame(data, columns=pd.Index(["algo1", "algo2"]))


@pytest.fixture
def dummy_features():
    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [1.2],
            [2.2],
            [3.2],
            [4.2],
            [5.2],
            [0.8],
            [1.8],
            [2.8],
            [3.8],
            [4.8],
            [1.5],
            [2.5],
            [3.5],
            [4.5],
            [5.5],
        ]
    )
    return pd.DataFrame(data, columns=pd.Index(["f1"]))


@pytest.fixture
def validate_predictions():
    def _validate(predictions):
        """
        Validates that predictions have the expected structure:
        - Length of predictions is 20.
        - Each value in predictions is a list.
        - Each list has a length of 1.
        """
        assert len(predictions) == 20
        for key in predictions:
            assert isinstance(predictions[key], list)
            assert len(predictions[key]) >= 1

    return _validate
