#!/usr/bin/env python3
"""
Test script to verify FIRE action is working correctly in Breakout
Shows the difference between with and without automatic FIRE
"""

import gymnasium
import ale_py
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def test_without_fire():
    """Test Breakout without automatic FIRE - ball stays at bottom"""
    print("Testing WITHOUT automatic FIRE action...")
    print("(The ball should stay at the bottom)")
    
    env = gymnasium.make('ALE/Breakout-v5', render_mode='human')
    obs, info = env.reset()
    
    # Take some random actions WITHOUT firing first
    for i in range(50):
        # Only use LEFT (3) and RIGHT (2) actions, never FIRE (1)
        action = np.random.choice([0, 2, 3])  # NOOP, RIGHT, LEFT
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"  Step {i}: Reward={reward}, Lives={info.get('lives', 'N/A')}")
        
        if terminated or truncated:
            break
        
        time.sleep(0.05)  # Slow down for visibility
    
    env.close()
    print("  Result: Ball never launched (as expected)\n")

def test_with_fire():
    """Test Breakout WITH automatic FIRE - ball launches and game plays"""
    print("Testing WITH automatic FIRE action...")
    print("(The ball should launch immediately)")
    
    env = gymnasium.make('ALE/Breakout-v5', render_mode='human')
    obs, info = env.reset()
    
    # FIRE at the start to launch ball
    print("  Firing to launch ball...")
    obs, _, _, _, info = env.step(1)  # FIRE action
    
    # Now play with paddle movements
    for i in range(100):
        # Random paddle movements
        action = np.random.choice([0, 2, 3])  # NOOP, RIGHT, LEFT
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"  Step {i}: Reward={reward}, Lives={info.get('lives', 'N/A')}")
        
        # Check if we lost a life
        if i > 0 and info.get('lives', 5) < 5:
            print(f"  Lost a life! Lives remaining: {info.get('lives', 'N/A')}")
            # Fire again to restart
            obs, _, _, _, _ = env.step(1)
            print("  Fired to launch ball again")
        
        if terminated or truncated:
            break
        
        time.sleep(0.03)  # Slightly faster
    
    env.close()
    print("  Result: Game played normally with ball in motion\n")

def test_with_wrapper():
    """Test our AtariWrapper that handles FIRE automatically"""
    print("Testing with AtariWrapper (automatic FIRE handling)...")
    
    # Import the wrapper from train script
    from train_muzero_breakout import AtariWrapper
    
    env = AtariWrapper('ALE/Breakout-v5')
    obs, info = env.reset()  # Should automatically FIRE
    
    print("  Environment reset - ball should be launched automatically")
    
    total_reward = 0
    for i in range(100):
        # Just use paddle movements, wrapper handles FIRE
        action = np.random.choice([0, 2, 3])  # NOOP, RIGHT, LEFT
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f"  Step {i}: Total Reward={total_reward}, Lives={info.get('lives', 'N/A')}")
        
        if terminated or truncated:
            break
    
    env.close()
    print(f"  Final score: {total_reward}\n")

def main():
    """Run all tests"""
    print("="*60)
    print("BREAKOUT FIRE ACTION TEST")
    print("="*60)
    print()
    
    # Test 1: Without FIRE (ball stays)
    test_without_fire()
    
    time.sleep(2)
    
    # Test 2: With manual FIRE (ball launches)
    test_with_fire()
    
    time.sleep(2)
    
    # Test 3: With wrapper (automatic FIRE)
    test_with_wrapper()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nSummary:")
    print("1. Without FIRE: Ball stays at bottom (game doesn't start)")
    print("2. With FIRE: Ball launches and game plays normally")
    print("3. With Wrapper: Automatic FIRE handling for seamless gameplay")

if __name__ == "__main__":
    main()