import os
import numpy as np
import pandas as pd

from asf.presolving.asap_v2 import ASAPv2
from asf.scenario.aslib_reader import read_aslib_scenario
from asf.selectors.performance_model import PerformanceModel
from asf.selectors.selector_pipeline import SelectorPipeline
from sklearn.ensemble import RandomForestRegressor


def test_asap_v2_basic():
    """Test basic functionality of ASAPv2"""
    print("\n" + "=" * 60)
    print("Testing ASAPv2 Basic Functionality")
    print("=" * 60 + "\n")
    
    # Create synthetic data
    np.random.seed(42)
    n_instances = 50
    n_algorithms = 4
    
    # Create feature data
    features = pd.DataFrame(
        np.random.randn(n_instances, 5),
        columns=[f'feature_{i}' for i in range(5)],
        index=[f'instance_{i}' for i in range(n_instances)]
    )
    
    # Create performance data (runtimes)
    base_times = np.random.exponential(scale=20, size=(n_instances, n_algorithms))
    performance = pd.DataFrame(
        base_times,
        columns=[f'algorithm_{i}' for i in range(n_algorithms)],
        index=features.index
    )
    
    # Split into train/test sets
    train_size = int(0.8 * n_instances)
    
    # Training data
    train_features = features.iloc[:train_size]
    train_performance = performance.iloc[:train_size]
    # Test data (completely separate!)
    test_features = features.iloc[train_size:]
    
    print(f"Created synthetic data: {n_instances} instances, {n_algorithms} algorithms")
    print(f"Training: {len(train_features)} instances")
    print(f"Testing: {len(test_features)} instances")
    print(f"Features shape: {features.shape}")
    print(f"Performance shape: {performance.shape}")
    
    # Test ASAPv2 initialization
    asap = ASAPv2(
        budget=100.0,
        presolver_cutoff=30.0,
        de_maxiter=10,
        de_popsize=5,
        verbosity=1
    )
    
    print(f"\nInitialized ASAPv2 with:")
    print(f"  Budget: {asap.budget}")
    print(f"  Presolver cutoff: {asap.presolver_cutoff}")
    
    # Fitting ASAPv2 on training data
    print("\nFitting ASAPv2 on training data...")
    asap.fit(train_features, train_performance)
    
    # Check results
    print(f"\nFit completed!")
    print(f"Final schedule: {asap.schedule}")
    
    # Test prediction on test data
    print(f"\nTesting prediction on {len(test_features)} new instances...")
    predictions = asap.predict(test_features)
    
    # Only show first 3 predictions, should always be the same schedule
    for instance_id, schedule in list(predictions.items())[:3]:
        print(f"  {instance_id}: {schedule}")
    
    # Verify all test instances get the same schedule (expected for ASAP)
    all_schedules = list(predictions.values())
    if len(set(str(schedule) for schedule in all_schedules)) == 1:
        print("✓ All instances get the same preschedule (expected for ASAP)")
    else:
        print("⚠️  Different schedules for different instances (unexpected)")
    
    print("\n✅ Basic functionality test passed!\n")
    return True


def test_asap_v2_with_aslib_scenario():
    """Test ASAPv2 with a real ASlib scenario"""
    print("\n" + "=" * 60)
    print("Testing ASAPv2 with ASlib Scenario")
    print("=" * 60 + "\n")
    
    # Path to ASlib scenarios (adjust as needed)
    scenario_path = "/home/schiller/asf/aslib_data/TSP-LION2015"
    
    if not os.path.exists(scenario_path):
        print(f"⚠️  ASlib scenario not found at {scenario_path}")
        print("Skipping ASlib scenario test")
        return False
    
    try:
        # Load ASlib scenario
        print(f"Loading ASlib scenario from: {scenario_path}")
        (
            features,
            performance,
            features_running_time,
            cv,
            feature_groups,
            maximize,
            budget,
        ) = read_aslib_scenario(scenario_path)
        
        print(f"Loaded scenario:")
        print(f"  Features shape: {features.shape}")
        print(f"  Performance shape: {performance.shape}")
        print(f"  Algorithms: {list(performance.columns)}")
        
        # Use a subset for faster testing
        subset_size = min(100, len(features))
        features_subset = features.iloc[:subset_size]
        performance_subset = performance.iloc[:subset_size]
        
        print(f"\nUsing subset of {subset_size} instances for testing")
        
        # Test ASAPv2 - REMOVED size_preschedule
        asap = ASAPv2(
            budget=budget,
            presolver_cutoff=min(budget / 3, 60.0),  # Use 1/3 of budget or 60s max
            de_maxiter=20,
            de_popsize=8,
            regularization_weight=0.01,
            verbosity=1
        )
        
        print(f"\nInitialized ASAPv2 with:")
        print(f"  Budget: {asap.budget}")
        print(f"  Presolver cutoff: {asap.presolver_cutoff}")

        print("\nFitting ASAPv2 on training data...")

        asap.fit(features_subset, performance_subset)
        
        # Test prediction
        test_features = features_subset.iloc[:10]
        predictions = asap.predict(test_features)
        print(f"\nGenerated schedules for {len(predictions)} instances")
        
        # Show one example schedule
        example_instance = list(predictions.keys())[0]
        example_schedule = predictions[example_instance]
        print(f"Example schedule for {example_instance}: {example_schedule}")
        
        print("\n✅ ASlib scenario test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ASlib scenario: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_asap_v2_in_selector_pipeline():
    """Test ASAPv2 integrated in a SelectorPipeline with meaningful evaluation"""
    # Does the selector pipeline work actually use the presolver?
    # Calls pre_solving, but does not use the presolver's schedule
    print("not implemented yet")


def test_asap_v2_edge_cases():
    """Test ASAPv2 edge cases"""
    print("\n" + "=" * 60)
    print("Testing ASAPv2 Edge Cases")
    print("=" * 60)
    
    # Test with single algorithm
    print("Test 1: Single algorithm")
    features = pd.DataFrame(np.random.randn(10, 2), columns=['f1', 'f2'])
    performance = pd.DataFrame(np.random.exponential(10, (10, 1)), columns=['alg1'])
    
    asap = ASAPv2(budget=30.0, presolver_cutoff=10.0, verbosity=1)
    asap.fit(features, performance)
    print(f"Single algorithm schedule: {asap.schedule}")
    
    # Test with two algorithms
    print("\nTest 2: Two algorithms")
    performance = pd.DataFrame(np.random.exponential(10, (10, 2)), columns=['alg1', 'alg2'])
    
    asap = ASAPv2(budget=30.0, presolver_cutoff=10.0, verbosity=1)
    asap.fit(features, performance)
    print(f"Two algorithm schedule: {asap.schedule}")
    
    # Test with many algorithms
    print("\nTest 3: Five algorithms")
    performance = pd.DataFrame(
        np.random.exponential(10, (10, 5)), 
        columns=['alg1', 'alg2', 'alg3', 'alg4', 'alg5']
    )
    
    asap = ASAPv2(budget=30.0, presolver_cutoff=10.0, verbosity=1)
    asap.fit(features, performance)
    print(f"Five algorithm schedule: {asap.schedule}")
    
    print("\n✅ Edge cases test completed!")
    return True


def test_asap_v2_parameters():
    """Test different parameter configurations"""
    print("\n" + "=" * 60)
    print("Testing ASAPv2 Parameter Variations")
    print("=" * 60)
    
    # Create test data
    np.random.seed(102)
    features = pd.DataFrame(np.random.randn(20, 3), columns=['f1', 'f2', 'f3'])
    performance = pd.DataFrame(
        np.random.exponential(10, (20, 4)), 
        columns=['alg1', 'alg2', 'alg3', 'alg4']
    )
    
    regularization_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    for i, reg_weight in enumerate(regularization_weights, 1):
        print(f"\nTest {i}: regularization weight={reg_weight}")
        
        asap = ASAPv2(
            budget=40.0,
            presolver_cutoff=15.0,
            de_maxiter=5,
            de_popsize=4,
            regularization_weight=reg_weight
        )
        
        asap.fit(features, performance)

        print(f"  Schedule: {asap.schedule}")

    
    print("\n✅ Parameter variations test completed!")
    return True


def run_all_tests():
    """Run all ASAPv2 tests"""
    print("Starting ASAPv2 Test Suite")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_asap_v2_basic),
        ("ASlib Scenario", test_asap_v2_with_aslib_scenario),
        ("SelectorPipeline Integration", test_asap_v2_in_selector_pipeline),
        ("Edge Cases", test_asap_v2_edge_cases),
        ("Parameter Variations", test_asap_v2_parameters),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "⚠️  FAILED"
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "❌ ERROR"
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        print(f"{result:<15} {test_name}")
    
    passed = sum(1 for result in results.values() if "PASSED" in result)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed or had errors")
    
    return results


if __name__ == "__main__":
    # You can run individual tests or all tests
    
    # Run all tests
    run_all_tests()
    
    # Or run individual tests:
    # test_asap_v2_basic()
    # test_asap_v2_with_aslib_scenario()