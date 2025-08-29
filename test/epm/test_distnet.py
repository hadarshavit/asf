"""
Tests for DistNet predictor implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def test_distnet_import():
    """Test that DistNet can be imported."""
    try:
        from asf.epm.distnet import DistNet
        assert DistNet is not None
    except ImportError as e:
        if "PyTorch is not available" in str(e):
            pytest.skip("PyTorch not available, skipping DistNet tests")
        else:
            raise


@pytest.mark.skipif(
    condition=not _torch_available(),
    reason="PyTorch not available"
)
class TestDistNet:
    """Test suite for DistNet predictor."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic data
        self.n_samples = 100
        self.n_features = 5
        
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        
        # Create target with some noise
        true_coeff = np.random.randn(self.n_features)
        self.y = pd.Series(
            self.X.values @ true_coeff + 0.1 * np.random.randn(self.n_samples)
        )
        
    def test_distnet_initialization(self):
        """Test DistNet initialization."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(
            hidden_sizes=[64, 32],
            dropout=0.1,
            activation='relu',
            num_components=3,
            epochs=5  # Small for testing
        )
        
        assert distnet.hidden_sizes == [64, 32]
        assert distnet.dropout == 0.1
        assert distnet.activation == 'relu'
        assert distnet.num_components == 3
        assert distnet.epochs == 5
        
    def test_distnet_fit_predict(self):
        """Test fitting and prediction."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(
            hidden_sizes=[32, 16],
            epochs=5,  # Small for testing
            batch_size=32
        )
        
        # Fit the model
        distnet.fit(self.X, self.y)
        
        # Make predictions
        predictions = distnet.predict(self.X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.y)
        assert not np.isnan(predictions).any()
        
    def test_distnet_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(
            hidden_sizes=[32, 16],
            epochs=5,
            num_components=3
        )
        
        distnet.fit(self.X, self.y)
        
        # Test uncertainty prediction
        results = distnet.predict_with_uncertainty(self.X)
        
        assert 'mean' in results
        assert 'variance' in results
        assert 'mixture_weights' in results
        assert 'component_means' in results
        assert 'component_variances' in results
        
        assert len(results['mean']) == len(self.y)
        assert len(results['variance']) == len(self.y)
        assert results['mixture_weights'].shape[0] == len(self.y)
        assert results['mixture_weights'].shape[1] == 3  # num_components
        
        # Check that variances are positive
        assert np.all(results['variance'] > 0)
        
        # Check that mixture weights sum to 1
        np.testing.assert_allclose(
            np.sum(results['mixture_weights'], axis=1), 
            1.0, 
            rtol=1e-5
        )
        
    def test_distnet_configuration_space(self):
        """Test configuration space generation."""
        try:
            from asf.epm.distnet import DistNet
            from ConfigSpace import ConfigurationSpace
            
            cs = DistNet.get_configuration_space()
            assert isinstance(cs, ConfigurationSpace)
            
            # Check that expected parameters are present
            param_names = [param.name for param in cs.get_hyperparameters()]
            expected_params = [
                'DistNet:hidden_size_1',
                'DistNet:hidden_size_2', 
                'DistNet:hidden_size_3',
                'DistNet:dropout',
                'DistNet:activation',
                'DistNet:use_batch_norm',
                'DistNet:num_components',
                'DistNet:learning_rate',
                'DistNet:batch_size',
                'DistNet:mixture_loss_weight'
            ]
            
            for param in expected_params:
                assert param in param_names
                
        except ImportError:
            pytest.skip("ConfigSpace not available")
            
    def test_distnet_from_configuration(self):
        """Test creating DistNet from configuration."""
        try:
            from asf.epm.distnet import DistNet
            
            config = {
                'DistNet:hidden_size_1': 128,
                'DistNet:hidden_size_2': 64,
                'DistNet:hidden_size_3': 32,
                'DistNet:dropout': 0.3,
                'DistNet:activation': 'tanh',
                'DistNet:use_batch_norm': False,
                'DistNet:num_components': 4,
                'DistNet:learning_rate': 0.01,
                'DistNet:batch_size': 128,
                'DistNet:mixture_loss_weight': 0.2
            }
            
            distnet_partial = DistNet.get_from_configuration(config)
            distnet = distnet_partial()
            
            assert distnet.hidden_sizes == [128, 64, 32]
            assert distnet.dropout == 0.3
            assert distnet.activation == 'tanh'
            assert distnet.use_batch_norm == False
            assert distnet.num_components == 4
            assert distnet.learning_rate == 0.01
            assert distnet.batch_size == 128
            assert distnet.mixture_loss_weight == 0.2
            
        except ImportError:
            pytest.skip("ConfigSpace not available")
            
    def test_distnet_save_load_limitation(self):
        """Test save/load functionality and its current limitation."""
        from asf.epm.distnet import DistNet
        import tempfile
        import os
        
        distnet = DistNet(epochs=2)
        distnet.fit(self.X, self.y)
        
        # Test saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            distnet.save(f.name)
            
            # Test that loading raises appropriate error due to missing input size
            distnet_new = DistNet()
            with pytest.raises(ValueError, match="Cannot load model without knowing input size"):
                distnet_new.load(f.name)
                
            # Clean up
            os.unlink(f.name)
            
    def test_distnet_error_conditions(self):
        """Test error conditions."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet()
        
        # Test prediction before fitting
        with pytest.raises(ValueError, match="Model must be fitted"):
            distnet.predict(self.X)
            
        with pytest.raises(ValueError, match="Model must be fitted"):
            distnet.predict_with_uncertainty(self.X)
            
        # Test sample weights not supported
        with pytest.raises(ValueError, match="Sample weights are not supported"):
            distnet.fit(self.X, self.y, sample_weight=np.ones(len(self.y)))


@pytest.mark.skipif(
    condition=_torch_available(),
    reason="PyTorch is available"
)
def test_distnet_import_error_when_torch_unavailable():
    """Test that DistNet raises ImportError when PyTorch is unavailable."""
    with patch.dict('sys.modules', {'torch': None}):
        from asf.epm.distnet import DistNet
        
        with pytest.raises(ImportError, match="PyTorch is not available"):
            DistNet()