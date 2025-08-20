#!/usr/bin/env python3
"""
Test Breakout manually to understand the game dynamics
"""

import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

def test_breakout():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    
    print("Testing Breakout dynamics...")
    print(f"Action space: {env.action_space}")
    print("Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT")
    
    # Test 1: What happens with no actions?
    print("\n--- Test 1: No actions (NOOP only) ---")
    obs, _ = env.reset()
    for i in range(100):
        obs, reward, done, _, _ = env.step(0)  # NOOP
        if reward > 0:
            print(f"  Got reward {reward} at step {i}")
        if done:
            print(f"  Game ended at step {i}")
            break
    print(f"  Result: Game doesn't start without FIRE")
    
    # Test 2: Fire to start
    print("\n--- Test 2: FIRE to start ---")
    obs, _ = env.reset()
    obs, reward, done, _, _ = env.step(1)  # FIRE
    print(f"  After FIRE: reward={reward}, done={done}")
    for i in range(50):
        obs, reward, done, _, _ = env.step(0)  # NOOP
        if reward > 0:
            print(f"  Got reward {reward} at step {i} (ball hit something)")
            break
        if done:
            print(f"  Game ended at step {i}")
            break
    
    # Test 3: Optimal simple strategy
    print("\n--- Test 3: Simple strategy (FIRE + alternating movement) ---")
    obs, _ = env.reset()
    obs, reward, done, _, _ = env.step(1)  # FIRE
    total_reward = reward
    
    for i in range(500):
        # Simple strategy: move right and left alternately
        action = 2 if (i // 20) % 2 == 0 else 3  # Switch every 20 steps
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if reward > 0:
            print(f"  Step {i}: Got reward {reward}, total={total_reward}")
        
        if done:
            break
    
    print(f"  Final reward with simple strategy: {total_reward}")
    
    # Test 4: Random agent as baseline
    print("\n--- Test 4: Random agent (proper) ---")
    obs, _ = env.reset()
    obs, reward, done, _, _ = env.step(1)  # FIRE first
    total_reward = reward
    
    for i in range(500):
        action = np.random.choice([0, 2, 3], p=[0.2, 0.4, 0.4])  # Mostly move, some NOOP
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if reward > 0:
            print(f"  Step {i}: Got reward {reward}, total={total_reward}")
        
        if done:
            break
    
    print(f"  Final reward with random agent: {total_reward}")
    
    env.close()

if __name__ == '__main__':
    test_breakout()