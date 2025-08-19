#!/usr/bin/env python3
"""
Play and record MuZero agent gameplay
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


def play_and_record(agent, num_seconds=10, fps=30, output_path=None, render_mode='human'):
    """
    Play the game and record video
    
    Args:
        agent: Trained MuZero agent
        num_seconds: Duration of video in seconds
        fps: Frames per second for video
        output_path: Path to save video
        render_mode: 'human' for display, 'rgb_array' for recording
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'./results/muzero_gameplay_{timestamp}.mp4'
    
    # Create environment with render mode
    env = gym.make(agent.config.env_name, render_mode='rgb_array')
    
    # Video writer setup
    frame = env.reset()[0]
    if hasattr(env, 'render'):
        frame = env.render()
    
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Recording gameplay to {output_path}")
    print(f"Video settings: {width}x{height} @ {fps}fps for {num_seconds}s")
    
    # Play game
    observation, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    start_time = time.time()
    max_frames = num_seconds * fps
    frames_recorded = 0
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    print("\nStarting gameplay...")
    
    while time.time() - start_time < num_seconds and frames_recorded < max_frames:
        # Preprocess observation for MuZero
        obs_tensor = agent.preprocess_observation(observation).to(agent.device)
        
        # Use MCTS to select action (no exploration)
        with torch.no_grad():
            mcts_result = agent.mcts.run(
                obs_tensor,
                agent.network,
                temperature=0,  # Deterministic
                add_exploration_noise=False
            )
        
        action = mcts_result['action']
        value = mcts_result['value']
        
        # Take action
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update metrics
        total_reward += reward
        current_episode_reward += reward
        current_episode_length += 1
        steps += 1
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frames_recorded += 1
        
        # Log progress every second
        elapsed = time.time() - start_time
        if int(elapsed) > int(elapsed - 1/fps) and elapsed > 0:
            print(f"  {int(elapsed)}s: Steps={steps}, Total Reward={total_reward:.1f}, "
                  f"Current Episode={current_episode_reward:.1f}, Value={value:.2f}")
        
        # Reset if episode ends
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            print(f"  Episode {len(episode_rewards)} completed: "
                  f"Reward={current_episode_reward:.1f}, Length={current_episode_length}")
            
            current_episode_reward = 0
            current_episode_length = 0
            
            if time.time() - start_time < num_seconds:
                observation, _ = env.reset()
                done = False
    
    # Cleanup
    out.release()
    env.close()
    
    # Final statistics
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Gameplay Recording Complete")
    print(f"{'='*60}")
    print(f"Video saved to: {output_path}")
    print(f"Duration: {duration:.1f}s ({frames_recorded} frames)")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.1f}")
    
    if episode_rewards:
        print(f"\nEpisode Statistics ({len(episode_rewards)} episodes):")
        print(f"  Mean reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"  Max reward: {np.max(episode_rewards):.1f}")
        print(f"  Min reward: {np.min(episode_rewards):.1f}")
        print(f"  Mean length: {np.mean(episode_lengths):.1f}")
    
    return output_path, episode_rewards


def main():
    parser = argparse.ArgumentParser(description='Play and record MuZero agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--game', type=str, default='Breakout',
                       help='Atari game name')
    parser.add_argument('--duration', type=int, default=10,
                       help='Recording duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--simulations', type=int, default=50,
                       help='Number of MCTS simulations')
    parser.add_argument('--no-video', action='store_true',
                       help='Play without recording video')
    
    args = parser.parse_args()
    
    # Create config
    config = MuZeroConfig()
    config.env_name = f'ALE/{args.game}-v5'
    config.num_simulations = args.simulations
    
    # Create and load agent
    print(f"Loading MuZero agent from {args.checkpoint}")
    agent = MuZeroAgent(config)
    agent.load_checkpoint(args.checkpoint)
    
    print(f"\nAgent loaded successfully:")
    print(f"  Games played: {agent.games_played}")
    print(f"  Training steps: {agent.training_steps}")
    print(f"  MCTS simulations: {config.num_simulations}")
    
    if args.no_video:
        # Just play without recording
        print("\nPlaying without recording...")
        eval_metrics = agent.evaluate(num_episodes=5, render=True)
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {eval_metrics['mean_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
        print(f"  Max reward: {eval_metrics['max_reward']:.1f}")
    else:
        # Play and record
        video_path, rewards = play_and_record(
            agent,
            num_seconds=args.duration,
            fps=args.fps,
            output_path=args.output
        )
        
        print(f"\n✅ Video successfully saved to: {video_path}")
        
        # Try to get video file size
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()