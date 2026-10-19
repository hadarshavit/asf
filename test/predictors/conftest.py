from typing import Any
import numpy as np
import pytest


@pytest.fixture
def classification_data() -> Any:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1])
    return X, y


@pytest.fixture
def regression_data() -> Any:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = X.sum(axis=1)
    return X, y
