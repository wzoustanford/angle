#!/usr/bin/env python3
"""
Analyze the action sequence from MuZero gameplay
"""

import torch
import numpy as np
import gymnasium as gym
import ale_py
from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent

gym.register_envs(ale_py)

# Action meanings for Breakout
ACTION_MEANINGS = {
    0: "NOOP (No Operation)",
    1: "FIRE (Start/Launch Ball)",
    2: "RIGHT (Move Paddle Right)",
    3: "LEFT (Move Paddle Left)"
}

def analyze_gameplay():
    # Load the trained agent
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 25
    
    agent = MuZeroAgent(config)
    checkpoint_path = './results/muzero_checkpoints/muzero_final_Breakout_20250815_181618.pth'
    agent.load_checkpoint(checkpoint_path)
    
    # Create environment
    env = gym.make('ALE/Breakout-v5')
    
    print("=" * 70)
    print("MUZERO AGENT ACTION SEQUENCE ANALYSIS")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Simulations per action: {config.num_simulations}")
    print("\nAction Meanings:")
    for action, meaning in ACTION_MEANINGS.items():
        print(f"  {action}: {meaning}")
    print("\n" + "=" * 70)
    print("ACTION SEQUENCE (10 seconds of gameplay):")
    print("=" * 70)
    
    # Reset environment
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    fps = 30
    target_frames = 10 * fps  # 10 seconds at 30fps
    
    # Track action sequences
    action_sequence = []
    action_counts = {i: 0 for i in range(4)}
    
    # Track consecutive actions
    last_action = None
    consecutive_count = 0
    
    print("\nStep | Action | Description                    | Reward | Total | Notes")
    print("-" * 80)
    
    for frame in range(target_frames):
        # Get action from agent
        obs_tensor = agent.preprocess_observation(obs).to(agent.device)
        
        with torch.no_grad():
            mcts_result = agent.mcts.run(
                obs_tensor,
                agent.network,
                temperature=0,
                add_exploration_noise=False
            )
        
        action = mcts_result['action']
        action_sequence.append(action)
        action_counts[action] += 1
        
        # Take action in environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track consecutive actions
        if action == last_action:
            consecutive_count += 1
        else:
            if last_action is not None and consecutive_count > 1:
                # Print summary of consecutive actions when they change
                pass  # Already printed individual steps
            consecutive_count = 1
            last_action = action
        
        # Print detailed info for key frames or when something happens
        if frame < 30 or frame % 30 == 0 or reward > 0 or action != 0:
            notes = ""
            if frame == 0:
                notes = "← Game Start"
            elif reward > 0:
                notes = f"← SCORE! (+{reward})"
            elif action == 1:
                notes = "← Firing ball!"
            elif action in [2, 3]:
                notes = "← Moving paddle!"
            
            print(f"{frame:4d} | {action:6d} | {ACTION_MEANINGS[action]:30s} | {reward:6.1f} | {total_reward:5.1f} | {notes}")
        
        if terminated or truncated:
            print(f"\n>>> Game ended at step {frame}")
            break
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS:")
    print("=" * 70)
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"\nAction Distribution:")
    for action in range(4):
        percentage = (action_counts[action] / step_count) * 100
        print(f"  {ACTION_MEANINGS[action]:30s}: {action_counts[action]:3d} times ({percentage:5.1f}%)")
    
    # Analyze patterns
    print(f"\n" + "=" * 70)
    print("BEHAVIORAL ANALYSIS:")
    print("=" * 70)
    
    # Check if agent ever fired
    if action_counts[1] == 0:
        print("⚠️  Agent NEVER fired the ball (Action 1) - Game never started!")
    else:
        print(f"✓  Agent fired {action_counts[1]} times")
    
    # Check if agent ever moved
    if action_counts[2] == 0 and action_counts[3] == 0:
        print("⚠️  Agent NEVER moved the paddle (Actions 2 or 3)")
    else:
        print(f"✓  Agent moved right {action_counts[2]} times, left {action_counts[3]} times")
    
    # Check for variety
    unique_actions = sum(1 for count in action_counts.values() if count > 0)
    if unique_actions == 1:
        print("⚠️  CRITICAL: Agent only uses 1 action - no gameplay diversity!")
    elif unique_actions == 2:
        print("⚠️  WARNING: Agent only uses 2 different actions")
    else:
        print(f"✓  Agent uses {unique_actions} different actions")
    
    # Analyze sequences
    print(f"\n" + "=" * 70)
    print("ACTION PATTERNS:")
    print("=" * 70)
    
    # Find longest consecutive sequence
    max_consecutive = 1
    current_consecutive = 1
    max_action = action_sequence[0]
    
    for i in range(1, len(action_sequence)):
        if action_sequence[i] == action_sequence[i-1]:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                max_action = action_sequence[i]
        else:
            current_consecutive = 1
    
    print(f"Longest consecutive sequence: {max_consecutive} times of {ACTION_MEANINGS[max_action]}")
    
    # Show first 100 actions as a compact string
    print(f"\nFirst 100 actions (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT):")
    action_string = ''.join(str(a) for a in action_sequence[:100])
    for i in range(0, min(100, len(action_string)), 50):
        print(f"  Steps {i:3d}-{min(i+49, len(action_string)-1):3d}: {action_string[i:i+50]}")
    
    env.close()

if __name__ == '__main__':
    analyze_gameplay()