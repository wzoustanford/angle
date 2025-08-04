#!/usr/bin/env python3
"""
Play script for loading and visualizing trained Space Invaders DQN model.
Loads checkpoints produced by space_invaders_dqn.py and displays gameplay.
"""

import torch
import numpy as np
import gymnasium as gym
import ale_py
import argparse
import os
import glob
from typing import Optional
import time

from space_invaders_dqn import DQN, AgentConfig, FrameStack

gym.register_envs(ale_py)

class SpaceInvadersPlayer:
    """Player class for loading and running trained Space Invaders DQN model"""
    
    def __init__(self, checkpoint_path: str, record_video: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load checkpoint with safe globals for custom classes
        torch.serialization.add_safe_globals([AgentConfig])
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']
        
        # Setup environment with appropriate rendering mode
        if record_video:
            self.env = gym.make(self.config.env_name, render_mode='rgb_array')
            self.video_path = record_video
            self.frames = []
        else:
            self.env = gym.make(self.config.env_name, render_mode='human')
            self.video_path = None
            self.frames = None
        
        self.n_actions = self.env.action_space.n
        
        # Frame preprocessing
        self.frame_stack = FrameStack(self.config.frame_stack)
        
        # Load trained network
        obs_shape = (self.config.frame_stack * 3, 210, 160)
        self.q_network = DQN(obs_shape, self.n_actions).to(self.device)
        self.q_network.load_state_dict(self.checkpoint['q_network_state_dict'])
        self.q_network.eval()
        
        print(f"Loaded checkpoint from episode {self.checkpoint['episode']}")
        print(f"Total training steps: {self.checkpoint['steps_done']}")
        print(f"Final epsilon: {self.checkpoint['epsilon']:.3f}")
    
    def select_action(self, state: np.ndarray, use_random: bool = False) -> int:
        """Select action using trained policy (greedy) or random"""
        if use_random:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def play_episodes(self, num_episodes: int = 5, fps: int = 30, 
                     use_random: bool = False, show_stats: bool = True):
        """Play multiple episodes with visualization"""
        episode_rewards = []
        episode_lengths = []
        
        frame_time = 1.0 / fps if fps > 0 else 0
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset environment
            obs, _ = self.env.reset()
            state = self.frame_stack.reset(obs)
            
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state, use_random=use_random)
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Record frame if making video
                if self.video_path:
                    frame = self.env.render()
                    if frame is not None:
                        self.frames.append(frame)
                
                # Update state
                state = self.frame_stack.append(next_obs)
                episode_reward += reward
                step_count += 1
                
                # Control frame rate
                if frame_time > 0:
                    time.sleep(frame_time)
                
                # Show real-time stats
                if show_stats and step_count % 100 == 0:
                    print(f"  Step {step_count}, Reward: {episode_reward}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            print(f"Episode {episode + 1} finished:")
            print(f"  Total Reward: {episode_reward}")
            print(f"  Steps: {step_count}")
        
        # Summary statistics
        print(f"\n=== Summary ({num_episodes} episodes) ===")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Best Reward: {np.max(episode_rewards)}")
        print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
        
        # Save video if recording
        if self.video_path and self.frames:
            self._save_video()
        
        self.env.close()
        return episode_rewards, episode_lengths
    
    def _save_video(self):
        """Save recorded frames as MP4 video"""
        try:
            import cv2
            print(f"\nSaving video to {self.video_path}...")
            
            if not self.frames:
                print("No frames to save!")
                return
            
            # Get frame dimensions
            height, width, channels = self.frames[0].shape
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (width, height))
            
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved successfully! ({len(self.frames)} frames)")
            
        except ImportError:
            print("Error: OpenCV (cv2) not installed. Cannot save video.")
            print("Install with: pip install opencv-python")
        except Exception as e:
            print(f"Error saving video: {e}")

def find_latest_checkpoint(checkpoint_dir: str = './checkpoints') -> Optional[str]:
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_step_*.pth'))
    if not checkpoint_files:
        return None
    
    # Sort by step number (extract from filename)
    def extract_step(filename):
        basename = os.path.basename(filename)
        step_str = basename.replace('checkpoint_step_', '').replace('.pth', '')
        return int(step_str)
    
    latest_checkpoint = max(checkpoint_files, key=extract_step)
    return latest_checkpoint

def list_checkpoints(checkpoint_dir: str = './checkpoints'):
    """List available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return []
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_step_*.pth'))
    if not checkpoint_files:
        print(f"No checkpoints found in '{checkpoint_dir}'")
        return []
    
    print(f"Available checkpoints in '{checkpoint_dir}':")
    for i, checkpoint in enumerate(sorted(checkpoint_files), 1):
        basename = os.path.basename(checkpoint)
        step_str = basename.replace('checkpoint_step_', '').replace('.pth', '')
        print(f"  {i}. {basename} (step {step_str})")
    
    return sorted(checkpoint_files)

def main():
    parser = argparse.ArgumentParser(description='Play trained Space Invaders DQN model')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       help='Path to checkpoint file (default: latest in ./checkpoints)')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                       help='Number of episodes to play (default: 5)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate for rendering (default: 30, 0 for unlimited)')
    parser.add_argument('--random', action='store_true',
                       help='Use random actions instead of trained policy')
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable real-time statistics display')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available checkpoints and exit')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory containing checkpoints (default: ./checkpoints)')
    parser.add_argument('--record', '-r', type=str,
                       help='Record gameplay to video file (e.g., gameplay.mp4) instead of live display')
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list:
        list_checkpoints(args.checkpoint_dir)
        return
    
    # Find checkpoint to use
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
            return
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoints found in '{args.checkpoint_dir}'")
            print("Use --list to see available checkpoints or specify --checkpoint")
            return
        print(f"Using latest checkpoint: {os.path.basename(checkpoint_path)}")
    
    try:
        # Create player and play
        player = SpaceInvadersPlayer(checkpoint_path, record_video=args.record)
        
        if args.record:
            print(f"Recording gameplay to: {args.record}")
        elif args.random:
            print("Playing with RANDOM actions (baseline comparison)")
        else:
            print("Playing with TRAINED policy")
        
        player.play_episodes(
            num_episodes=args.episodes,
            fps=args.fps,
            use_random=args.random,
            show_stats=not args.no_stats
        )
        
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    except Exception as e:
        print(f"Error during playback: {e}")

if __name__ == "__main__":
    main()