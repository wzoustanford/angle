#!/usr/bin/env python3
"""
Test script to verify gymnasium integration with EfficientZero
"""

import gymnasium as gym
import numpy as np
from core.utils import make_atari, WarpFrame, EpisodicLifeEnv, TimeLimit
from config.atari.env_wrapper import AtariWrapper

def test_basic_gymnasium():
    """Test basic gymnasium functionality"""
    print("Testing basic gymnasium environment creation...")
    try:
        env = gym.make("PongNoFrameskip-v4")
        print(f"✓ Created environment: {env.spec.id}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset returns tuple: obs shape={obs.shape}, info={info}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step returns 5 values: obs shape={obs.shape}, reward={reward}, terminated={terminated}, truncated={truncated}")
        
        env.close()
        print("✓ Basic gymnasium test passed!")
        return True
    except Exception as e:
        print(f"✗ Basic gymnasium test failed: {e}")
        return False

def test_efficientzero_wrappers():
    """Test EfficientZero wrappers with gymnasium"""
    print("\nTesting EfficientZero wrappers...")
    try:
        # Create environment with EfficientZero wrappers
        env_name = "PongNoFrameskip-v4"
        env = make_atari(env_name, skip=4, max_episode_steps=1000)
        print(f"✓ Created environment with make_atari: {env_name}")
        
        # Apply wrappers
        env = EpisodicLifeEnv(env)
        print("✓ Applied EpisodicLifeEnv wrapper")
        
        env = WarpFrame(env, width=96, height=96, grayscale=True)
        print("✓ Applied WarpFrame wrapper")
        
        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        print(f"✓ Reset successful: obs shape={obs.shape}")
        
        # Test step
        for _ in range(10):
            action = env.env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            if done:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
        print(f"✓ Step successful: obs shape={obs.shape}, reward={reward}, done={done}")
        
        env.close()
        print("✓ EfficientZero wrappers test passed!")
        return True
    except Exception as e:
        print(f"✗ EfficientZero wrappers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_atari_wrapper():
    """Test AtariWrapper with gymnasium"""
    print("\nTesting AtariWrapper...")
    try:
        # Create environment
        env_name = "PongNoFrameskip-v4"
        env = make_atari(env_name, skip=4, max_episode_steps=1000)
        env = WarpFrame(env, width=96, height=96, grayscale=True)
        
        # Wrap with AtariWrapper
        wrapped_env = AtariWrapper(env, discount=0.997, cvt_string=False, seed=42)
        print(f"✓ Created AtariWrapper")
        
        # Test reset with seed
        obs = wrapped_env.reset()
        print(f"✓ Reset successful: obs type={type(obs)}")
        
        # Test step
        action = 0  # NOOP action
        obs, reward, done, info = wrapped_env.step(action)
        print(f"✓ Step successful: obs type={type(obs)}, reward={reward}, done={done}")
        
        # Test legal actions
        legal_actions = wrapped_env.legal_actions()
        print(f"✓ Legal actions: {len(legal_actions)} actions available")
        
        wrapped_env.close()
        print("✓ AtariWrapper test passed!")
        return True
    except Exception as e:
        print(f"✗ AtariWrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("EfficientZero Gymnasium Migration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Basic Gymnasium", test_basic_gymnasium()))
    results.append(("EfficientZero Wrappers", test_efficientzero_wrappers()))
    results.append(("AtariWrapper", test_atari_wrapper()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✓ All tests passed! Gymnasium migration successful.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())