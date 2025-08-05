#!/usr/bin/env python3
"""
Simple test runner for distributed RL components.
Can be run from the RL folder to execute unit tests.
"""

import unittest
import sys
import os
import argparse

def run_test_file(test_file):
    """Run a specific test file"""
    print(f"Running tests from {test_file}...")
    
    # Import and run the specific test
    if test_file == 'buffer':
        from tests.test_distributed_buffer import TestDistributedReplayBuffer, TestDistributedFrameStack
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestDistributedReplayBuffer))
        suite.addTest(unittest.makeSuite(TestDistributedFrameStack))
        
    elif test_file == 'env':
        from tests.test_parallel_env_manager import TestEnvironmentWorker, TestParallelEnvironmentManager, TestIntegration
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestEnvironmentWorker))
        suite.addTest(unittest.makeSuite(TestParallelEnvironmentManager))
        suite.addTest(unittest.makeSuite(TestIntegration))
        
    elif test_file == 'agent':
        from tests.test_distributed_dqn_agent import TestDistributedDQNAgent, TestDistributedAgentConfig, TestIntegrationDistributed
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestDistributedDQNAgent))
        suite.addTest(unittest.makeSuite(TestDistributedAgentConfig))
        suite.addTest(unittest.makeSuite(TestIntegrationDistributed))
        
    else:
        print(f"Unknown test file: {test_file}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """Run all tests using the main test runner"""
    print("Running all distributed RL tests...")
    
    # Import and run the comprehensive test runner
    from tests.run_all_tests import run_tests_by_category
    return run_tests_by_category()

def run_quick_test():
    """Run quick smoke test"""
    print("Running quick smoke test...")
    
    from tests.run_all_tests import run_quick_smoke_test
    return run_quick_smoke_test()

def main():
    parser = argparse.ArgumentParser(description='Run distributed RL tests')
    parser.add_argument('test_type', nargs='?', default='all',
                       choices=['all', 'quick', 'buffer', 'env', 'agent'],
                       help='Type of test to run (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("Distributed RL Test Runner")
    print("=" * 40)
    
    if args.test_type == 'quick':
        success = run_quick_test()
    elif args.test_type == 'all':
        success = run_all_tests()
    else:
        success = run_test_file(args.test_type)
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())