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
