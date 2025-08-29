#!/usr/bin/env python3
"""
Example demonstrating DistNet usage for algorithm runtime distribution prediction.

This example shows how to use DistNet for predicting algorithm runtime distributions
with uncertainty quantification, specifically designed for algorithm selection scenarios.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# For demonstration, we'll create synthetic runtime data
def create_synthetic_runtime_data(n_samples=500, n_features=10, n_algorithms=3):
    """Create synthetic algorithm runtime data with realistic characteristics."""
    np.random.seed(42)
    
    # Generate instance features (e.g., problem size, complexity measures)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = np.random.uniform(0, 10, (n_samples, n_features))
    
    # Create realistic runtime patterns for different algorithms
    algorithm_names = [f'algorithm_{i}' for i in range(n_algorithms)]
    
    runtimes = {}
    for i, alg in enumerate(algorithm_names):
        # Each algorithm has different runtime characteristics
        # Base runtime depends on features with some algorithm-specific scaling
        base_runtime = np.exp(
            0.1 * X[:, 0] +  # Problem size effect
            0.05 * X[:, 1] + # Complexity effect
            np.random.normal(0, 0.3, n_samples) + # Random noise
            i * 0.5  # Algorithm-specific offset
        )
        
        # Add some feature interactions specific to each algorithm
        if i == 0:  # Algorithm 0 is sensitive to feature 2
            base_runtime *= (1 + 0.2 * X[:, 2])
        elif i == 1:  # Algorithm 1 has quadratic complexity on feature 3
            base_runtime *= (1 + 0.01 * X[:, 3] ** 2)
        
        # Ensure positive runtimes and add some outliers
        runtimes[alg] = np.maximum(base_runtime, 0.01)
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    runtime_df = pd.DataFrame(runtimes)
    
    return X_df, runtime_df

def demonstrate_distnet_runtime():
    """Demonstrate DistNet for algorithm runtime prediction."""
    print("DistNet Algorithm Runtime Distribution Prediction Example")
    print("=" * 60)
    
    # Create synthetic runtime data
    print("1. Creating synthetic algorithm runtime data...")
    X, runtime_df = create_synthetic_runtime_data()
    print(f"   Dataset shape: X={X.shape}, runtimes={runtime_df.shape}")
    print(f"   Algorithms: {list(runtime_df.columns)}")
    print(f"   Runtime statistics:")
    print(f"   {runtime_df.describe()}")
    
    try:
        from asf.epm.distnet import DistNet
        print("\n✓ DistNet imported successfully")
        
        # We'll demonstrate on one algorithm's runtime
        algorithm = runtime_df.columns[0]
        y = runtime_df[algorithm]
        
        print(f"\n2. Training DistNet for {algorithm} runtime prediction...")
        
        # Create and configure DistNet
        distnet = DistNet(
            hidden_sizes=[64, 32, 16],
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
        print("   Training in progress...")
        distnet.fit(X_train, y_train)
        print("✓ DistNet training completed")
        
        # Make predictions
        print("\n3. Making runtime predictions...")
        predictions = distnet.predict(X_test)
        distribution_results = distnet.predict_distribution(X_test)
        
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Distribution parameters computed for {len(distribution_results['mean'])} samples")
        
        # Calculate performance metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        # Relative error (important for runtime prediction)
        mape = np.mean(np.abs((predictions - y_test) / (y_test + 1e-6))) * 100
        
        print(f"\n4. Performance Metrics:")
        print(f"   Mean Squared Error: {mse:.4f}")
        print(f"   Mean Absolute Error: {mae:.4f}")
        print(f"   Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Analyze runtime distribution predictions
        print(f"\n5. Runtime Distribution Analysis:")
        print(f"   Predicted mean runtime: {np.mean(distribution_results['mean']):.4f}")
        print(f"   Predicted median runtime: {np.mean(distribution_results['median']):.4f}")
        print(f"   Average predicted std: {np.mean(distribution_results['std']):.4f}")
        print(f"   Actual mean runtime: {np.mean(y_test):.4f}")
        
        # Show prediction intervals
        pred_mean = distribution_results['mean']
        pred_std = distribution_results['std']
        
        # 95% prediction intervals for log-normal distribution
        # For log-normal, we can compute quantiles
        lower_95 = distribution_results['median'] * np.exp(-1.96 * np.sqrt(distribution_results['log_variance']))
        upper_95 = distribution_results['median'] * np.exp(1.96 * np.sqrt(distribution_results['log_variance']))
        
        coverage = np.mean((y_test >= lower_95) & (y_test <= upper_95))
        print(f"   95% prediction interval coverage: {coverage:.2%}")
        
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
        
        print("\n✅ DistNet runtime prediction demonstration completed successfully!")
        
        # Optionally create visualization if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\n7. Creating runtime prediction visualization...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Predictions vs actual
            ax1.scatter(y_test, predictions, alpha=0.6, s=20)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Runtime')
            ax1.set_ylabel('Predicted Runtime')
            ax1.set_title('Runtime Predictions vs Actual')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True)
            
            # Plot 2: Prediction intervals
            sorted_indices = np.argsort(y_test)
            y_sorted = y_test.iloc[sorted_indices]
            pred_sorted = predictions[sorted_indices]
            lower_sorted = lower_95[sorted_indices]
            upper_sorted = upper_95[sorted_indices]
            
            ax2.plot(range(len(y_sorted)), y_sorted, 'o', label='Actual', markersize=3)
            ax2.plot(range(len(pred_sorted)), pred_sorted, 's', label='Predicted', markersize=3)
            ax2.fill_between(range(len(y_sorted)), lower_sorted, upper_sorted, 
                           alpha=0.3, label='95% Prediction Interval')
            ax2.set_xlabel('Test Sample (sorted by actual runtime)')
            ax2.set_ylabel('Runtime')
            ax2.set_title('Prediction Intervals')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Predicted std vs actual error
            actual_errors = np.abs(predictions - y_test)
            ax3.scatter(distribution_results['std'], actual_errors, alpha=0.6, s=20)
            ax3.set_xlabel('Predicted Standard Deviation')
            ax3.set_ylabel('Actual Absolute Error')
            ax3.set_title('Uncertainty vs Error')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.grid(True)
            
            # Plot 4: Runtime distribution histogram
            ax4.hist(y_test, bins=30, alpha=0.7, label='Actual Runtimes', density=True)
            ax4.hist(predictions, bins=30, alpha=0.7, label='Predicted Runtimes', density=True)
            ax4.set_xlabel('Runtime')
            ax4.set_ylabel('Density')
            ax4.set_title('Runtime Distribution Comparison')
            ax4.set_xscale('log')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('/tmp/distnet_runtime_demo.png', dpi=150, bbox_inches='tight')
            print("✓ Visualization saved to /tmp/distnet_runtime_demo.png")
            
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
    demonstrate_distnet_runtime()