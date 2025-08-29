"""
Tests for DistNet algorithm runtime distribution prediction.
"""

import pytest
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ConfigSpace import ConfigurationSpace
    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


def test_distnet_import():
    """Test that DistNet can be imported properly."""
    from asf.epm.distnet import DistNet
    assert DistNet is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDistNet:
    """Test suite for DistNet runtime distribution predictor."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic runtime data
        n_samples = 100
        n_features = 5
        
        # Features (instance characteristics)
        self.X = pd.DataFrame(
            np.random.uniform(0, 5, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Runtime targets (log-normal distributed)
        log_runtimes = (0.5 * self.X['feature_0'] + 
                       0.3 * self.X['feature_1'] + 
                       np.random.normal(0, 0.5, n_samples))
        self.y = pd.Series(np.exp(log_runtimes))
        
        # Train/test split
        split = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]

    def test_distnet_initialization(self):
        """Test DistNet initialization with various parameters."""
        from asf.epm.distnet import DistNet
        
        # Default initialization
        distnet = DistNet()
        assert distnet.hidden_sizes == [128, 64, 32]
        assert distnet.dropout == 0.1
        assert distnet.activation == 'relu'
        
        # Custom initialization
        distnet_custom = DistNet(
            hidden_sizes=[64, 32],
            dropout=0.2,
            activation='tanh',
            learning_rate=0.01
        )
        assert distnet_custom.hidden_sizes == [64, 32]
        assert distnet_custom.dropout == 0.2
        assert distnet_custom.activation == 'tanh'
        assert distnet_custom.learning_rate == 0.01

    def test_distnet_fit_predict(self):
        """Test basic fit and predict functionality."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(epochs=5, hidden_sizes=[32, 16])  # Small model for testing
        
        # Fit the model
        distnet.fit(self.X_train, self.y_train)
        
        # Check model was created
        assert distnet.model is not None
        assert distnet.optimizer is not None
        assert distnet.scaler is not None
        assert distnet.feature_scaler is not None
        
        # Make predictions
        predictions = distnet.predict(self.X_test)
        
        # Check predictions
        assert len(predictions) == len(self.X_test)
        assert all(predictions > 0)  # Runtimes should be positive
        assert not np.any(np.isnan(predictions))

    def test_distnet_predict_distribution(self):
        """Test distribution prediction functionality."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(epochs=5, hidden_sizes=[32, 16])
        distnet.fit(self.X_train, self.y_train)
        
        # Get distribution predictions
        dist_results = distnet.predict_distribution(self.X_test)
        
        # Check all required keys are present
        required_keys = ['mean', 'variance', 'median', 'std', 'log_mean', 'log_variance']
        for key in required_keys:
            assert key in dist_results
            assert len(dist_results[key]) == len(self.X_test)
            assert not np.any(np.isnan(dist_results[key]))
        
        # Check that variance and std are positive
        assert all(dist_results['variance'] > 0)
        assert all(dist_results['std'] > 0)
        
        # Check that mean and median are positive (for runtime)
        assert all(dist_results['mean'] > 0)
        assert all(dist_results['median'] > 0)

    def test_distnet_sample_weights_warning(self):
        """Test that sample weights issue a warning."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet(epochs=2, hidden_sizes=[16])
        
        # Should warn about sample weights
        with pytest.warns(UserWarning):
            distnet.fit(self.X_train, self.y_train, sample_weight=np.ones(len(self.X_train)))

    def test_distnet_save_load(self):
        """Test model saving and loading."""
        from asf.epm.distnet import DistNet
        import tempfile
        import os
        
        distnet = DistNet(epochs=3, hidden_sizes=[32, 16])
        distnet.fit(self.X_train, self.y_train)
        
        # Make predictions before saving
        pred_before = distnet.predict(self.X_test[:5])
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            distnet.save(f.name)
            
            # Create new instance and load
            distnet_loaded = DistNet()
            distnet_loaded.load(f.name)
            
            # Make predictions after loading
            pred_after = distnet_loaded.predict(self.X_test[:5])
            
            # Predictions should be identical
            np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)
            
            # Clean up
            os.unlink(f.name)

    def test_distnet_prediction_errors(self):
        """Test prediction errors when model is not fitted."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet()
        
        # Should raise error for unfitted model
        with pytest.raises(ValueError, match="Model must be fitted"):
            distnet.predict(self.X_test)
            
        with pytest.raises(ValueError, match="Model must be fitted"):
            distnet.predict_distribution(self.X_test)

    @pytest.mark.skipif(not CONFIGSPACE_AVAILABLE, reason="ConfigSpace not available")
    def test_distnet_configuration_space(self):
        """Test configuration space generation."""
        from asf.epm.distnet import DistNet
        from ConfigSpace import ConfigurationSpace
        
        cs = DistNet.get_configuration_space()
        
        # Check that it's a valid ConfigurationSpace
        assert isinstance(cs, ConfigurationSpace)
        
        # Check expected parameters
        param_names = [param.name for param in cs.get_hyperparameters()]
        expected_params = [
            'DistNet:hidden_size_1', 'DistNet:hidden_size_2', 'DistNet:hidden_size_3',
            'DistNet:dropout', 'DistNet:activation', 'DistNet:learning_rate', 'DistNet:batch_size'
        ]
        
        for param in expected_params:
            assert param in param_names

    @pytest.mark.skipif(not CONFIGSPACE_AVAILABLE, reason="ConfigSpace not available")
    def test_distnet_from_configuration(self):
        """Test creating DistNet from configuration."""
        from asf.epm.distnet import DistNet
        
        cs = DistNet.get_configuration_space()
        config = cs.sample_configuration()
        
        # Create DistNet from configuration
        distnet_factory = DistNet.get_from_configuration(config)
        distnet = distnet_factory()
        
        # Check that it's a valid DistNet instance
        assert isinstance(distnet, DistNet)
        
        # Check that parameters were set correctly
        assert distnet.hidden_sizes == [
            config['DistNet:hidden_size_1'],
            config['DistNet:hidden_size_2'],
            config['DistNet:hidden_size_3']
        ]
        assert distnet.dropout == config['DistNet:dropout']
        assert distnet.activation == config['DistNet:activation']


@pytest.mark.skipif(TORCH_AVAILABLE, reason="PyTorch is available")
class TestDistNetWithoutTorch:
    """Test DistNet behavior when PyTorch is not available."""

    def test_distnet_import_error(self):
        """Test that DistNet raises ImportError when PyTorch is not available."""
        from asf.epm.distnet import DistNet
        
        with pytest.raises(ImportError, match="PyTorch is not available"):
            DistNet()

    def test_distnet_methods_import_error(self):
        """Test that DistNet methods raise ImportError when PyTorch is not available."""
        from asf.epm.distnet import DistNet
        
        distnet = DistNet.__new__(DistNet)  # Create without calling __init__
        
        with pytest.raises(ImportError, match="PyTorch is not available"):
            distnet.fit(None, None)
            
        with pytest.raises(ImportError, match="PyTorch is not available"):
            distnet.predict(None)