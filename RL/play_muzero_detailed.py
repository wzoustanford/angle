#!/usr/bin/env python3
"""
Play and record MuZero agent gameplay with detailed action logging
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
import ale_py
from PIL import Image
import cv2
import os
import time
from datetime import datetime

from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent

# Register Atari environments
gym.register_envs(ale_py)

# Action meanings for Breakout
ACTION_MEANINGS = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT"
}

ACTION_DESCRIPTIONS = {
    0: "No Operation (stand still)",
    1: "Fire/Launch Ball",
    2: "Move Paddle Right →",
    3: "Move Paddle Left ←"
}


def play_and_record_detailed(agent, num_seconds=10, fps=30, output_path=None):
    """
    Play the game and record video with detailed action logging
    
    Args:
        agent: Trained MuZero agent
        num_seconds: Duration of video in seconds
        fps: Frames per second for video
        output_path: Path to save video
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'./results/muzero_gameplay_{timestamp}.mp4'
    
    # Create environment
    env = gym.make(agent.config.env_name, render_mode='rgb_array')
    
    print("=" * 80)
    print("MUZERO GAMEPLAY WITH DETAILED ACTION SEQUENCE")
    print("=" * 80)
    print(f"Model: {agent.games_played} games played, {agent.training_steps} training steps")
    print(f"Recording to: {output_path}")
    print(f"Duration: {num_seconds} seconds at {fps} FPS")
    print("\nAction Meanings:")
    for action, desc in ACTION_DESCRIPTIONS.items():
        print(f"  {action} ({ACTION_MEANINGS[action]:5s}): {desc}")
    print("=" * 80)
    
    # Video writer setup
    observation, _ = env.reset()
    frame = env.render()
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Video settings: {width}x{height} @ {fps}fps for {num_seconds}s")
    
    # Gameplay variables
    total_frames = num_seconds * fps
    total_reward = 0
    episode_reward = 0
    step_count = 0
    
    # Action tracking
    action_sequence = []
    action_counts = {i: 0 for i in range(4)}
    consecutive_actions = []
    last_action = None
    consecutive_count = 0
    
    # For Breakout, start with FIRE
    if 'Breakout' in agent.config.env_name:
        print("\n🎮 Starting Breakout - Firing ball to begin game...")
        for _ in range(3):
            observation, reward, terminated, truncated, _ = env.step(1)  # FIRE
            if terminated or truncated:
                observation, _ = env.reset()
        print("   Ball launched! Game started.\n")
    
    print("FRAME | ACTION | DESCRIPTION                      | REWARD | TOTAL | VALUE")
    print("-" * 80)
    
    start_time = time.time()
    
    for frame_num in range(total_frames):
        # Get action from agent (no MCTS for speed)
        obs_tensor = agent.preprocess_observation(observation).to(agent.device)
        
        with torch.no_grad():
            # Direct network inference without MCTS
            output = agent.network.initial_inference(obs_tensor.unsqueeze(0))
            policy_logits = output['policy_logits'].squeeze(0)
            
            # Deterministic action selection (no exploration)
            action = policy_logits.argmax().item()
            value = output['value'].item()
        
        # Track actions
        action_sequence.append(action)
        action_counts[action] += 1
        
        # Track consecutive actions
        if action == last_action:
            consecutive_count += 1
        else:
            if last_action is not None and consecutive_count >= 5:
                consecutive_actions.append((last_action, consecutive_count))
            consecutive_count = 1
            last_action = action
        
        # Take action
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        total_reward += reward
        step_count += 1
        
        # Log important frames
        if frame_num < 10 or frame_num % 30 == 0 or reward > 0 or action != 0:
            action_name = ACTION_MEANINGS[action]
            action_desc = ACTION_DESCRIPTIONS[action]
            print(f"{frame_num:5d} | {action_name:6s} | {action_desc:32s} | {reward:6.1f} | {total_reward:5.1f} | {value:6.2f}")
            
            if reward > 0:
                print(f"      >>> 🎯 SCORE! Brick destroyed! (+{reward} points)")
        
        # Render and save frame
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        # Reset if episode ends
        if terminated or truncated:
            print(f"\n      Episode ended at frame {frame_num}. Score: {episode_reward}")
            observation, _ = env.reset()
            episode_reward = 0
            
            # Fire again for Breakout
            if 'Breakout' in agent.config.env_name and frame_num < total_frames - 10:
                for _ in range(3):
                    observation, _, terminated, truncated, _ = env.step(1)
                    if terminated or truncated:
                        observation, _ = env.reset()
    
    # Clean up
    out.release()
    env.close()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("GAMEPLAY SUMMARY")
    print("=" * 80)
    print(f"Total frames: {total_frames}")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"Recording time: {elapsed:.1f}s")
    
    print(f"\nAction Distribution:")
    for action in range(4):
        percentage = (action_counts[action] / step_count) * 100 if step_count > 0 else 0
        print(f"  {ACTION_MEANINGS[action]:5s} ({ACTION_DESCRIPTIONS[action]:25s}): {action_counts[action]:3d} times ({percentage:5.1f}%)")
    
    # Analyze patterns
    print(f"\nAction Patterns:")
    if consecutive_actions:
        print("  Long consecutive sequences:")
        for action, count in sorted(consecutive_actions, key=lambda x: x[1], reverse=True)[:3]:
            print(f"    - {count} consecutive {ACTION_MEANINGS[action]} actions")
    
    # Check for problems
    print(f"\nBehavioral Analysis:")
    if action_counts[1] == 0:
        print("  ⚠️ WARNING: Never fired! Ball was never launched properly.")
    else:
        print(f"  ✅ Fired {action_counts[1]} times")
    
    if action_counts[2] == 0 and action_counts[3] == 0:
        print("  ⚠️ WARNING: Never moved paddle!")
    else:
        print(f"  ✅ Moved paddle: RIGHT {action_counts[2]} times, LEFT {action_counts[3]} times")
    
    unique_actions = sum(1 for count in action_counts.values() if count > 0)
    print(f"  📊 Action diversity: {unique_actions}/4 different actions used")
    
    # Show action sequence sample
    print(f"\nFirst 50 actions taken:")
    for i in range(0, min(50, len(action_sequence)), 10):
        seq_slice = action_sequence[i:i+10]
        seq_str = ''.join([ACTION_MEANINGS[a][0] for a in seq_slice])  # First letter of each action
        print(f"  Steps {i:3d}-{i+9:3d}: {seq_str} ", end="")
        print(f"({''.join([ACTION_MEANINGS[a] + ' ' for a in seq_slice[:3]])}...)")
    
    print(f"\n✅ Video saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Play MuZero agent with detailed logging')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--game', type=str, default='Breakout',
                        help='Game to play')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--simulations', type=int, default=1,
                        help='MCTS simulations (1 for no MCTS)')
    
    args = parser.parse_args()
    
    # Load configuration and agent
    config = MuZeroConfig()
    config.env_name = f'ALE/{args.game}-v5'
    config.num_simulations = args.simulations
    
    print(f"Loading MuZero agent from {args.checkpoint}")
    agent = MuZeroAgent(config)
    agent.load_checkpoint(args.checkpoint)
    print(f"Agent loaded successfully:")
    print(f"  Games played: {agent.games_played}")
    print(f"  Training steps: {agent.training_steps}")
    print(f"  MCTS simulations: {config.num_simulations}")
    
    # Play and record
    output_path = play_and_record_detailed(
        agent,
        num_seconds=args.duration,
        fps=args.fps,
        output_path=args.output
    )
    
    print(f"\n🎮 Gameplay recording complete!")


if __name__ == '__main__':
    main()