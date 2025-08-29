import numpy as np
import pytest
import pandas as pd
from asf.selectors import SNNAP, ISAC


def test_snnap_edge_cases():
    """Test SNNAP with edge cases."""
    # Create small dataset
    features = pd.DataFrame({
        'f1': [1, 2, 3, 4],
        'f2': [2, 3, 4, 5]
    }, index=[f'inst_{i}' for i in range(4)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30, 40],
        'algo2': [15, 25, 25, 35],
        'algo3': [12, 18, 28, 38]
    }, index=features.index)
    
    # Test with k larger than dataset
    selector = SNNAP(k=10)  # k > number of instances
    selector.fit(features, performance)
    predictions = selector.predict(features)
    
    # Check predictions format
    assert isinstance(predictions, dict)
    assert len(predictions) == len(features)
    for inst_name, pred in predictions.items():
        assert isinstance(pred, list)
        assert len(pred) == 1
        assert isinstance(pred[0], tuple)
        assert len(pred[0]) == 2
        assert pred[0][0] in performance.columns
        

def test_isac_edge_cases():
    """Test ISAC with edge cases."""
    # Create small dataset
    features = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5],
        'f2': [2, 3, 4, 5, 6]
    }, index=[f'inst_{i}' for i in range(5)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30, 40, 50],
        'algo2': [15, 25, 25, 35, 45]
    }, index=features.index)
    
    # Test clustering mode with more clusters than instances
    selector = ISAC(mode="clustering", n_clusters=10)
    selector.fit(features, performance)
    predictions = selector.predict(features)
    
    # Check predictions format
    assert isinstance(predictions, dict)
    assert len(predictions) == len(features)
    for inst_name, pred in predictions.items():
        assert isinstance(pred, list)
        assert len(pred) == 1
        assert isinstance(pred[0], tuple)
        assert len(pred[0]) == 2
        assert pred[0][0] in performance.columns


def test_snnap_maximization():
    """Test SNNAP with maximization objective."""
    features = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 3, 4]
    }, index=[f'inst_{i}' for i in range(3)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30],
        'algo2': [15, 25, 25]
    }, index=features.index)
    
    selector = SNNAP(k=2, maximize=True)
    selector.fit(features, performance)
    predictions = selector.predict(features)
    
    assert len(predictions) == len(features)


def test_isac_maximization():
    """Test ISAC with maximization objective."""
    features = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 3, 4]
    }, index=[f'inst_{i}' for i in range(3)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30],
        'algo2': [15, 25, 25]
    }, index=features.index)
    
    selector = ISAC(mode="regression", maximize=True)
    selector.fit(features, performance)
    predictions = selector.predict(features)
    
    assert len(predictions) == len(features)


def test_snnap_different_strategies():
    """Test SNNAP with different weight and selection strategies."""
    features = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5],
        'f2': [2, 3, 4, 5, 6]
    }, index=[f'inst_{i}' for i in range(5)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30, 40, 50],
        'algo2': [15, 25, 25, 35, 45],
        'algo3': [12, 22, 32, 42, 52]
    }, index=features.index)
    
    # Test all combinations
    strategies = [
        ("uniform", "voting"),
        ("uniform", "best_performance"),
        ("distance", "voting"),
        ("distance", "best_performance")
    ]
    
    for weight_strategy, algo_strategy in strategies:
        selector = SNNAP(
            k=3, 
            weight_strategy=weight_strategy,
            algorithm_selection_strategy=algo_strategy
        )
        selector.fit(features, performance)
        predictions = selector.predict(features)
        assert len(predictions) == len(features)


def test_isac_fallback_strategies():
    """Test ISAC fallback strategies."""
    features = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 3, 4]
    }, index=[f'inst_{i}' for i in range(3)])
    
    performance = pd.DataFrame({
        'algo1': [10, 20, 30],
        'algo2': [15, 25, 25]
    }, index=features.index)
    
    # Test different fallback strategies
    for fallback in ["best_overall", "random"]:
        selector = ISAC(
            mode="clustering", 
            n_clusters=2, 
            distance_threshold=0.1,  # Very small threshold to trigger fallback
            fallback_strategy=fallback
        )
        selector.fit(features, performance)
        predictions = selector.predict(features)
        assert len(predictions) == len(features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])