import pytest
import pandas as pd
import numpy as np
from src.transforms.target_transforms import TargetTransformer

def test_difference_transform():
    series = pd.Series([10, 15, 14, 20, 25])
    transformer = TargetTransformer(method="difference")
    
    transformed = transformer.fit_transform(series)
    assert len(transformed) == 4
    assert transformed.iloc[0] == 5
    assert transformed.iloc[1] == -1
    
    # Test inverse
    restored = transformer.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(series.iloc[1:].values, restored.values)
    
    # Test round trip helper
    assert transformer.round_trip_test(series)

def test_log_transform():
    series = pd.Series([10, 15, 14, 20, 25])
    transformer = TargetTransformer(method="log")
    
    transformed = transformer.fit_transform(series)
    restored = transformer.inverse_transform(transformed)
    
    np.testing.assert_array_almost_equal(series.values, restored.values)
    assert transformer.round_trip_test(series)

def test_box_cox_transform():
    series = pd.Series([10, 15, 14, 20, 25])
    transformer = TargetTransformer(method="box_cox")

    transformed = transformer.fit_transform(series)
    restored = transformer.inverse_transform(transformed)

    np.testing.assert_array_almost_equal(series.values, restored.values)
    assert transformer.round_trip_test(series)

def test_inverse_boxcox_clips_outside_negative_lambda_domain():
    from src.transforms.target_transforms import clip_to_boxcox_domain

    series = pd.Series([1.0, 2.0, 3.0, 4.0, 8.0])
    transformer = TargetTransformer(method="box_cox")
    transformer.fit(series)
    transformer._boxcox_lambda = -0.5
    transformer._boxcox_shift = 0.0

    wild = pd.Series([0.0, 1.5, 10.0, 100.0])
    restored = transformer.inverse_transform(wild)
    assert np.isfinite(restored.to_numpy(dtype=float)).all()

    bound = -1.0 / -0.5
    clipped = clip_to_boxcox_domain(wild.to_numpy(dtype=float), -0.5)
    assert np.all(clipped < bound)


def test_box_cox_adds_one_to_every_point():
    series = pd.Series([0.0, 1.0, 2.0, 4.0])
    transformer = TargetTransformer(method="box_cox")

    transformed = transformer.fit_transform(series)
    restored = transformer.inverse_transform(transformed)

    assert transformer._boxcox_shift == 1.0
    from scipy.stats import boxcox
    expected = boxcox(series.to_numpy(dtype=float) + 1.0, lmbda=transformer._boxcox_lambda)
    np.testing.assert_allclose(transformed.to_numpy(dtype=float), expected)
    assert transformed.iloc[0] != transformed.iloc[1]
    np.testing.assert_array_almost_equal(series.values, restored.values)
    assert transformer.round_trip_test(series)


def test_box_cox_lambda_is_refit_on_each_train_window():
    rng = np.random.default_rng(0)
    train = pd.Series(rng.uniform(1.0, 10.0, size=40))
    validation = pd.Series(rng.uniform(500.0, 800.0, size=10))
    window_train = pd.concat([train, validation.iloc[:5]], ignore_index=True)

    on_train = TargetTransformer(method="box_cox").fit(train)
    on_train_plus_validation = TargetTransformer(method="box_cox").fit(
        pd.concat([train, validation], ignore_index=True)
    )
    assert on_train._boxcox_lambda != pytest.approx(on_train_plus_validation._boxcox_lambda)
    assert on_train._boxcox_fit_size == len(train)

    # transform() keeps the training λ when it sees the validation tail.
    frozen = on_train._boxcox_lambda
    on_train.transform(pd.concat([train, validation], ignore_index=True))
    assert on_train._boxcox_lambda == frozen

    # The next expanding window estimates a new λ from its own train.
    on_train.fit(window_train)
    assert on_train._boxcox_lambda != pytest.approx(frozen)
    assert on_train._boxcox_fit_size == len(window_train)


def test_constant_series_falls_back_from_boxcox_to_log1p():
    zeros = pd.Series(np.zeros(12))
    transformer = TargetTransformer(method="box_cox")
    restored = transformer.inverse_transform(transformer.fit_transform(zeros))
    assert transformer.method == "log"
    np.testing.assert_array_almost_equal(zeros.values, restored.values)

    flat = pd.Series(np.full(8, 5.0))
    positive = TargetTransformer(method="box_cox")
    positive.fit(flat)
    assert positive.method == "log"
    assert positive.round_trip_test(flat)
