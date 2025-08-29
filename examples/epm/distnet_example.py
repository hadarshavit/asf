#!/usr/bin/env python3
"""
Example demonstrating DistNet usage for deep probability estimation.

This example shows how to use DistNet for probabilistic regression
with uncertainty quantification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# For demonstration, we'll create synthetic data even if PyTorch isn't available
def create_synthetic_data(n_samples=200, noise_level=0.2):
    """Create synthetic regression data with heteroscedastic noise."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.uniform(-3, 3, (n_samples, 2))
    
    # Create target with heteroscedastic noise (noise depends on input)
    true_function = lambda x: np.sin(x[:, 0]) + 0.5 * x[:, 1]**2
    y_mean = true_function(X)
    
    # Noise level depends on the absolute value of the function
    noise_scale = noise_level * (1 + np.abs(y_mean))
    y = y_mean + np.random.normal(0, noise_scale)
    
    return pd.DataFrame(X, columns=['x1', 'x2']), pd.Series(y)

def demonstrate_distnet():
    """Demonstrate DistNet capabilities."""
    print("DistNet Deep Probability Estimation Example")
    print("=" * 50)
    
    # Create synthetic data
    print("1. Creating synthetic data...")
    X, y = create_synthetic_data()
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    
    try:
        from asf.epm.distnet import DistNet
        print("✓ DistNet imported successfully")
        
        # Create and configure DistNet
        print("\n2. Creating DistNet model...")
        distnet = DistNet(
            hidden_sizes=[64, 32, 16],
            num_components=3,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        print("✓ DistNet model created")
        
        # Split data for training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train the model
        print("\n3. Training DistNet...")
        distnet.fit(X_train, y_train)
        print("✓ DistNet training completed")
        
        # Make predictions
        print("\n4. Making predictions...")
        predictions = distnet.predict(X_test)
        uncertainty_results = distnet.predict_with_uncertainty(X_test)
        
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Uncertainties computed for {len(uncertainty_results['mean'])} samples")
        
        # Calculate performance metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        print(f"\n5. Performance Metrics:")
        print(f"   Mean Squared Error: {mse:.4f}")
        print(f"   Mean Absolute Error: {mae:.4f}")
        
        # Analyze uncertainty
        mean_uncertainty = np.mean(uncertainty_results['variance'])
        print(f"   Mean Predicted Variance: {mean_uncertainty:.4f}")
        
        # Show mixture information
        mixture_weights = uncertainty_results['mixture_weights']
        print(f"   Mixture components shape: {mixture_weights.shape}")
        print(f"   Average mixture entropy: {np.mean(-np.sum(mixture_weights * np.log(mixture_weights + 1e-8), axis=1)):.4f}")
        
        print("\n6. Testing configuration space...")
        try:
            from ConfigSpace import ConfigurationSpace
            cs = DistNet.get_configuration_space()
            print(f"✓ Configuration space has {len(cs.get_hyperparameters())} parameters")
            
            # Test sampling a configuration
            config = cs.sample_configuration()
            distnet_from_config = DistNet.get_from_configuration(config)()
            print("✓ DistNet can be created from sampled configuration")
            
        except ImportError:
            print("⚠ ConfigSpace not available, skipping hyperparameter optimization features")
        
        print("\n✅ DistNet demonstration completed successfully!")
        
        # Optionally create visualization if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\n7. Creating visualization...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot predictions vs actual
            ax1.scatter(y_test, predictions, alpha=0.6)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title('Predictions vs Actual')
            ax1.grid(True)
            
            # Plot uncertainty vs error
            errors = np.abs(predictions - y_test)
            uncertainties = np.sqrt(uncertainty_results['variance'])
            ax2.scatter(uncertainties, errors, alpha=0.6)
            ax2.set_xlabel('Predicted Uncertainty (σ)')
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Uncertainty vs Error')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('/tmp/distnet_demo.png', dpi=150, bbox_inches='tight')
            print("✓ Visualization saved to /tmp/distnet_demo.png")
            
        else:
            print("⚠ Matplotlib not available, skipping visualization")
        
    except ImportError as e:
        if "PyTorch is not available" in str(e):
            print("⚠ PyTorch not available. DistNet requires PyTorch to function.")
            print("  Install with: pip install torch")
        else:
            print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_distnet()