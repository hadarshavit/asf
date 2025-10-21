import numpy as np

from asf.preprocessing.quadratic_expansion import expand_features


def test_quadratic_expansion_shapes_and_values():
    # X with 3 features -> expanded has 3 + 6 = 9 columns
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
        ]
    )
    X_exp = expand_features(X)

    assert X_exp.shape == (2, 9)

    # First k columns unchanged
    assert np.allclose(X_exp[:, :3], X)

    # Pairwise products in order (j<=l):
    # (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    expected_products_row0 = [
        1.0 * 1.0,
        1.0 * 2.0,
        1.0 * 3.0,
        2.0 * 2.0,
        2.0 * 3.0,
        3.0 * 3.0,
    ]
    expected_products_row1 = [
        0.5 * 0.5,
        0.5 * -1.0,
        0.5 * 4.0,
        (-1.0) * (-1.0),
        (-1.0) * 4.0,
        4.0 * 4.0,
    ]

    assert np.allclose(X_exp[0, 3:], expected_products_row0)
    assert np.allclose(X_exp[1, 3:], expected_products_row1)


def test_quadratic_expansion_single_feature():
    X = np.array([[2.0], [3.0]])
    X_exp = expand_features(X)
    # k=1 -> m = 1 + 1 = 2
    assert X_exp.shape == (2, 2)
    assert np.allclose(X_exp[:, 0], X[:, 0])
    # product (0,0)
    assert np.allclose(X_exp[:, 1], X[:, 0] ** 2)
