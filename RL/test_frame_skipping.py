#!/usr/bin/env python3
"""
Test the impact of frame skipping on Breakout gameplay
Compares performance with and without frame skipping
"""

import gymnasium
import ale_py
import numpy as np
import cv2
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

class AtariWrapperNoSkip:
    """Breakout wrapper WITHOUT frame skipping - more responsive"""
    
    def __init__(self, env_name='ALE/Breakout-v5', frame_stack=4):
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        obs, info = self.env.reset()
        # Fire to start
        obs, _, _, _, info = self.env.step(1)
        self._last_info = info
        
        # Initialize frame stack
        processed = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """NO FRAME SKIPPING - action executes once"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if we lost a life
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        lives_after = info.get('lives', 0)
        if lives_before > 0 and lives_after < lives_before and lives_after > 0:
            obs, _, _, _, _ = self.env.step(1)  # FIRE
        
        self._last_info = info
        
        # Add new frame
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        return self._get_stacked_frames(), reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _get_stacked_frames(self):
        return np.array(self.frames, dtype=np.float32)


class AtariWrapperWithSkip:
    """Breakout wrapper WITH frame skipping (original)"""
    
    def __init__(self, env_name='ALE/Breakout-v5', frame_stack=4, frame_skip=4):
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frame_stack)
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        obs, info = self.env.reset()
        # Fire to start
        obs, _, _, _, info = self.env.step(1)
        self._last_info = info
        
        # Initialize frame stack
        processed = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """WITH FRAME SKIPPING - action repeats frame_skip times"""
        total_reward = 0
        done = False
        
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Check if we lost a life
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        lives_after = info.get('lives', 0)
        if lives_before > 0 and lives_after < lives_before and lives_after > 0:
            obs, _, _, _, _ = self.env.step(1)  # FIRE
        
        self._last_info = info
        
        # Add new frame
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _get_stacked_frames(self):
        return np.array(self.frames, dtype=np.float32)


def test_random_agent(env, num_episodes=5, verbose=True):
    """Test environment with random agent"""
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 1000:
            # Random action (but avoid FIRE during gameplay)
            action = np.random.choice([0, 2, 3])  # NOOP, LEFT, RIGHT
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            print(f"  Episode {ep+1}: Reward={total_reward:.0f}, Steps={steps}")
    
    return episode_rewards, episode_lengths


def test_simple_ai(env, num_episodes=5, verbose=True):
    """Test with simple paddle-following AI"""
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 1000:
            # Simple heuristic: move paddle toward center with some randomness
            if steps % 10 < 5:
                action = 2  # RIGHT
            else:
                action = 3  # LEFT
            
            # Add some randomness
            if np.random.random() < 0.1:
                action = np.random.choice([0, 2, 3])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            print(f"  Episode {ep+1}: Reward={total_reward:.0f}, Steps={steps}")
    
    return episode_rewards, episode_lengths


def main():
    """Compare frame skipping vs no frame skipping"""
    
    print("="*60)
    print("FRAME SKIPPING COMPARISON TEST")
    print("="*60)
    
    # Test 1: With frame skipping (traditional)
    print("\n1. WITH Frame Skipping (4 frames per action):")
    print("-" * 40)
    env_skip = AtariWrapperWithSkip(frame_skip=4)
    
    print("Random agent:")
    rewards_skip_random, lengths_skip_random = test_random_agent(env_skip, num_episodes=3)
    
    print("\nSimple AI:")
    rewards_skip_ai, lengths_skip_ai = test_simple_ai(env_skip, num_episodes=3)
    
    env_skip.close()
    
    # Test 2: Without frame skipping (more responsive)
    print("\n2. WITHOUT Frame Skipping (1 frame per action):")
    print("-" * 40)
    env_no_skip = AtariWrapperNoSkip()
    
    print("Random agent:")
    rewards_no_skip_random, lengths_no_skip_random = test_random_agent(env_no_skip, num_episodes=3)
    
    print("\nSimple AI:")
    rewards_no_skip_ai, lengths_no_skip_ai = test_simple_ai(env_no_skip, num_episodes=3)
    
    env_no_skip.close()
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("\nRandom Agent:")
    print(f"  With Skip:    Avg Reward = {np.mean(rewards_skip_random):.1f}, Avg Length = {np.mean(lengths_skip_random):.0f}")
    print(f"  Without Skip: Avg Reward = {np.mean(rewards_no_skip_random):.1f}, Avg Length = {np.mean(lengths_no_skip_random):.0f}")
    
    print("\nSimple AI:")
    print(f"  With Skip:    Avg Reward = {np.mean(rewards_skip_ai):.1f}, Avg Length = {np.mean(lengths_skip_ai):.0f}")
    print(f"  Without Skip: Avg Reward = {np.mean(rewards_no_skip_ai):.1f}, Avg Length = {np.mean(lengths_no_skip_ai):.0f}")
    
    print("\n" + "="*60)
    print("IMPLICATIONS FOR MUZERO")
    print("="*60)
    
    print("\nFrame Skipping PROS:")
    print("  ✓ 4x faster training (fewer decisions needed)")
    print("  ✓ Matches original DQN/MuZero papers")
    print("  ✓ Reduces computational cost")
    
    print("\nFrame Skipping CONS:")
    print("  ✗ Less precise control (can miss the ball)")
    print("  ✗ Harder dynamics to learn (4-step jumps)")
    print("  ✗ May limit maximum performance")
    
    print("\nRECOMMENDATION:")
    print("  Start with frame_skip=4 for faster initial learning")
    print("  Reduce to frame_skip=2 or 1 for final fine-tuning")
    print("  Or make it adaptive based on game state")


if __name__ == "__main__":
    main()