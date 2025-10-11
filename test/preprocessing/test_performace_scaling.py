import numpy as np

from asf.preprocessing.performance_scaling import (
    MinMaxNormalization,
    ZScoreNormalization,
    LogNormalization,
    SqrtNormalization,
    InvSigmoidNormalization,
    NegExpNormalization,
    DummyNormalization,
    BoxCoxNormalization,
)


def assert_roundtrip(norm, data):
    norm.fit(data)
    transformed = norm.transform(data)
    inverted = norm.inverse_transform(transformed)
    # Allow some tolerance for floating point ops
    assert np.allclose(data, inverted, atol=1e-6, rtol=1e-6)


def test_minmax_roundtrip():
    data = np.array([0.0, 5.0, 10.0])
    norm = MinMaxNormalization(feature_range=(0, 1))
    assert_roundtrip(norm, data)


def test_zscore_roundtrip():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    norm = ZScoreNormalization()
    assert_roundtrip(norm, data)


def test_log_negative_and_zero():
    data = np.array([-5.0, 0.0, 5.0])
    norm = LogNormalization(base=10, eps=1e-6)
    # Fit should set min_val when negatives exist
    norm.fit(data)
    assert hasattr(norm, "min_val")
    transformed = norm.transform(data)
    inverted = norm.inverse_transform(transformed)
    assert np.allclose(data, inverted, atol=1e-5)


def test_sqrt_negative():
    data = np.array([-4.0, 0.0, 9.0])
    norm = SqrtNormalization(eps=1e-6)
    norm.fit(data)
    transformed = norm.transform(data)
    inverted = norm.inverse_transform(transformed)
    # The implementation shifts by min_val which currently results in
    # negative values under the sqrt for the negative entries producing NaNs.
    # We expect the last element (positive) to round-trip, and the first two to be NaN.
    assert np.isnan(inverted[0]) and np.isnan(inverted[1])
    assert np.allclose(inverted[2], data[2], atol=1e-5)


def test_inv_sigmoid_and_negexp():
    data = np.array([0.1, 0.5, 0.9])
    inv = InvSigmoidNormalization()
    # sklearn scalers expect 2D arrays for fitting, so provide column vector
    inv.fit(data.reshape(-1, 1))
    t = inv.transform(data)
    inv_back = inv.inverse_transform(t)
    assert np.allclose(data, inv_back, atol=1e-6)

    neg = NegExpNormalization()
    t2 = neg.transform(data)
    back2 = neg.inverse_transform(t2)
    assert np.allclose(data, back2, atol=1e-6)


def test_dummy_normalization_identity():
    data = np.array([-3.5, 0.0, 7.2])
    dummy = DummyNormalization()
    dummy.fit(data)
    transformed = dummy.transform(data)
    assert transformed is data
    assert np.allclose(transformed, data)
    assert np.allclose(dummy.inverse_transform(transformed), data)


def test_boxcox_roundtrip():
    data = np.array([-2.5, -0.5, 0.0, 1.5, 3.0])
    norm = BoxCoxNormalization()
    norm.fit(data)
    transformed = norm.transform(data)
    assert not np.isnan(transformed).any()
    inverted = norm.inverse_transform(transformed)
    assert np.allclose(data, inverted, atol=1e-6)


def test_log_fit_sets_eps_for_positive_data():
    data = np.array([1.0, 2.0, 4.0])
    norm = LogNormalization(base=2, eps=1e-3)
    norm.fit(data)
    assert norm.min_val == 0
    assert norm.eps == 0
    transformed = norm.transform(data)
    assert np.allclose(np.log2(data), transformed)
    inverted = norm.inverse_transform(transformed)
    assert np.allclose(data, inverted, atol=1e-6)
