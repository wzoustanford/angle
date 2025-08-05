#!/usr/bin/env python3
"""
Test runner for all distributed RL components.
Runs comprehensive unit tests to ensure the distributed algorithm works correctly.
"""

import unittest
import sys
import os
import time

# Add current directory to path for imports when run from RL folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import test modules
from tests.test_distributed_buffer import TestDistributedReplayBuffer, TestDistributedFrameStack
from tests.test_parallel_env_manager import TestEnvironmentWorker, TestParallelEnvironmentManager, TestIntegration
from tests.test_distributed_dqn_agent import TestDistributedDQNAgent, TestDistributedAgentConfig, TestIntegrationDistributed


def create_test_suite():
    """Create a comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add distributed buffer tests
    suite.addTest(unittest.makeSuite(TestDistributedReplayBuffer))
    suite.addTest(unittest.makeSuite(TestDistributedFrameStack))
    
    # Add parallel environment tests
    suite.addTest(unittest.makeSuite(TestEnvironmentWorker))
    suite.addTest(unittest.makeSuite(TestParallelEnvironmentManager))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Add distributed agent tests
    suite.addTest(unittest.makeSuite(TestDistributedDQNAgent))
    suite.addTest(unittest.makeSuite(TestDistributedAgentConfig))
    suite.addTest(unittest.makeSuite(TestIntegrationDistributed))
    
    return suite


def run_tests_by_category():
    """Run tests by category with progress reporting"""
    
    print("=" * 80)
    print("DISTRIBUTED REINFORCEMENT LEARNING TESTS")
    print("=" * 80)
    
    # Test categories
    test_categories = [
        ("Distributed Buffer Tests", [TestDistributedReplayBuffer, TestDistributedFrameStack]),
        ("Parallel Environment Tests", [TestEnvironmentWorker, TestParallelEnvironmentManager]),
        ("Distributed Agent Tests", [TestDistributedDQNAgent, TestDistributedAgentConfig]),
        ("Integration Tests", [TestIntegration, TestIntegrationDistributed])
    ]
    
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for category_name, test_classes in test_categories:
        print(f"\n{'-' * 60}")
        print(f"Running {category_name}")
        print(f"{'-' * 60}")
        
        category_suite = unittest.TestSuite()
        for test_class in test_classes:
            category_suite.addTest(unittest.makeSuite(test_class))
        
        # Run tests for this category
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        start_time = time.time()
        result = runner.run(category_suite)
        end_time = time.time()
        
        # Report category results
        print(f"\n{category_name} Results:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"  Failed: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures)
        total_errors += len(result.errors)
        
        # Print details of failures and errors
        if result.failures:
            print(f"\nFailures in {category_name}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        if result.errors:
            print(f"\nErrors in {category_name}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_passed + total_failed + total_errors}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    
    if total_failed == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! The distributed RL system is working correctly.")
        return True
    else:
        print(f"\n❌ {total_failed + total_errors} tests failed. Please check the output above.")
        return False


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("Running Quick Smoke Test...")
    print("-" * 40)
    
    try:
        # Test imports
        from model.distributed_buffer import DistributedReplayBuffer
        from model.parallel_env_manager import ParallelEnvironmentManager
        from model.distributed_dqn_agent import DistributedDQNAgent
        from config.DistributedAgentConfig import DistributedAgentConfig
        print("✓ All imports successful")
        
        # Test basic instantiation
        config = DistributedAgentConfig(num_workers=1, memory_size=100)
        buffer = DistributedReplayBuffer(100)
        print("✓ Basic object creation successful")
        
        # Test buffer operations
        import numpy as np
        buffer.push(np.random.rand(4, 84, 84), 0, 1.0, np.random.rand(4, 84, 84), False, 0)
        assert len(buffer) == 1
        print("✓ Buffer operations working")
        
        print("\n🚀 Smoke test passed! System appears to be working.")
        return True
        
    except Exception as e:
        print(f"\n💥 Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run distributed RL tests')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke test only')
    parser.add_argument('--category', type=str, help='Run specific category (buffer, env, agent, integration)')
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_smoke_test()
        sys.exit(0 if success else 1)
    
    # Run full test suite
    success = run_tests_by_category()
    sys.exit(0 if success else 1)