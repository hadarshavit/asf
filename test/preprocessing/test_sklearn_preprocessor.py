import numpy as np
import pandas as pd

from asf.preprocessing.sklearn_preprocessor import get_default_preprocessor


def test_get_default_preprocessor_imputes_and_encodes():
    df = pd.DataFrame(
        {
            "color": ["red", None, "blue"],
            "size": [1.0, 2.5, np.nan],
        }
    )

    preprocessor = get_default_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    expected_columns = [
        "cat__color_blue",
        "cat__color_red",
        "cat__color_None",
        "cont__size",
    ]
    assert transformed.columns.tolist() == expected_columns

    expected_values = np.array(
        [
            [0.0, 1.0, 0.0, -1.22474487],
            [0.0, 0.0, 1.0, 1.22474487],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(transformed.to_numpy(), expected_values, atol=1e-6)


def test_get_default_preprocessor_with_explicit_selectors():
    df = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "other_cat": ["foo", "bar"],
            "size": [1.0, 2.0],
            "other_num": [10, 20],
        }
    )

    preprocessor = get_default_preprocessor(
        categorical_features=["color"],
        numerical_features=["size"],
    )
    transformed = preprocessor.fit_transform(df)

    assert transformed.columns.tolist() == [
        "cat__color_blue",
        "cat__color_red",
        "cont__size",
    ]

    expected = np.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(transformed.to_numpy(), expected, atol=1e-6)
