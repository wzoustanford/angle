#!/usr/bin/env python3
"""
Generate 10-second video of MuZero playing Breakout
"""

import gymnasium
import ale_py
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import cv2
from collections import deque
import os
import warnings
warnings.filterwarnings('ignore')

class AtariWrapper:
    """
    Wrapper for Atari environments to handle preprocessing
    """
    def __init__(self, env_name='ALE/Breakout-v5', frame_stack=4):
        """
        Initialize Atari environment with preprocessing
        """
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        
        # Get action space info
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        
        # Observation shape after preprocessing
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        """Reset environment and return stacked frames"""
        obs, info = self.env.reset()
        
        # Take FIRE action to start the game (action 1 in Breakout)
        # This launches the ball at the beginning of each life
        obs, _, _, _, info = self.env.step(1)  # FIRE action
        
        # Store info for tracking lives
        self._last_info = info
        
        # Preprocess frame
        processed = self._preprocess_frame(obs)
        
        # Initialize frame stack
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """Take a step and return stacked frames"""
        # Repeat action for frame skipping (typically 4 frames)
        total_reward = 0
        done = False
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        
        for _ in range(4):  # Frame skip
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Check if we lost a life (ball went out)
        lives_after = info.get('lives', 0)
        if lives_before > 0 and lives_after < lives_before and lives_after > 0:
            # Lost a life but game continues - fire to launch ball again
            obs, _, _, _, _ = self.env.step(1)  # FIRE action
        
        self._last_info = info
        
        # Preprocess and add to frame stack
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def _preprocess_frame(self, frame):
        """
        Preprocess a single frame
        - Convert to grayscale
        - Resize to 84x84
        - Normalize to [0, 1]
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _get_stacked_frames(self):
        """Get stacked frames as numpy array"""
        return np.array(self.frames, dtype=np.float32)


def create_breakout_10sec_video():
    """Create exactly 10-second video of MuZero playing Breakout"""
    
    checkpoint_path = 'muzero_breakout_best.pt'
    video_path = 'muzero_breakout_10sec.mp4'
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("Training a quick model first (this will take a few minutes)...")
        
        # Quick training
        from train_muzero_breakout import train_muzero_breakout
        muzero, rewards = train_muzero_breakout(
            num_episodes=30,        # Very quick training
            num_simulations=10,     # Minimal simulations
            batch_size=32,
            learning_rate=1e-3,
            save_every=30,
            print_every=5,
            max_moves_per_episode=500,
            frame_stack=4
        )
        print("✅ Quick training complete!")
    
    print(f"\n📹 Creating 10-second video from {checkpoint_path}...")
    
    # Create environment
    env = AtariWrapper('ALE/Breakout-v5', frame_stack=4)
    
    # Initialize MuZero and load model
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=20,  # Balance speed and quality
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        muzero.load_checkpoint(checkpoint_path)
        print("✅ Loaded best model")
    
    # Target: 10 seconds at 30 fps = 300 frames
    target_frames = 300
    fps = 30
    
    print(f"Target: {target_frames} frames at {fps} fps")
    
    # Play and record
    all_frames = []
    obs, _ = env.reset()
    done = False
    total_score = 0
    total_steps = 0
    lives = 5  # Breakout starts with 5 lives
    episode_count = 1
    
    # Capture initial frame
    frame = env.render()
    all_frames.append(frame)
    
    while len(all_frames) < target_frames:
        # Use MCTS to select action
        action_probs = muzero.run_mcts(obs)
        
        # Select action (greedy for best performance)
        action = np.argmax(action_probs)
        
        # You could also use sampling for variety:
        # action = np.random.choice(len(action_probs), p=action_probs)
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Capture frame
        frame = env.render()
        all_frames.append(frame)
        
        total_score += reward
        total_steps += 1
        
        # Check if lost a life (can detect from reward pattern or info)
        if reward < 0:  # Lost a life in some Atari games
            lives -= 1
            if lives > 0:
                print(f"  Lost a life! Lives remaining: {lives}")
        
        # If game ended and we need more frames, start a new game
        if done and len(all_frames) < target_frames - 10:
            print(f"  Episode {episode_count}: Score={total_score:.0f}, Steps={total_steps}")
            episode_count += 1
            
            # Add transition frames
            for _ in range(5):
                if len(all_frames) < target_frames:
                    all_frames.append(np.zeros_like(frame))
            
            # Reset for new game
            obs, _ = env.reset()
            done = False
            total_score = 0
            total_steps = 0
            lives = 5
    
    env.close()
    
    # Trim to exactly target frames
    all_frames = all_frames[:target_frames]
    
    # Save video
    print(f"\n💾 Saving video to {video_path}...")
    
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    duration = len(all_frames) / fps
    
    print("\n" + "="*60)
    print("✅ Video created successfully!")
    print("="*60)
    print(f"📹 Video: {video_path}")
    print(f"⏱️  Duration: {duration:.1f} seconds ({len(all_frames)} frames)")
    print(f"🎮 Episodes played: {episode_count}")
    print(f"🏆 Final score: {total_score:.0f}")
    print(f"📊 Total steps: {total_steps}")
    print("\n🎬 Play the video with: ffplay " + video_path)
    
    return video_path


def create_comparison_video():
    """Create comparison video: Random vs Trained MuZero"""
    
    print("Creating comparison video: Random vs MuZero...")
    
    # Create environment
    env = AtariWrapper('ALE/Breakout-v5', frame_stack=4)
    
    # First: Random agent (5 seconds)
    print("\n1️⃣ Recording random agent...")
    random_frames = []
    obs, _ = env.reset()
    
    for _ in range(150):  # 5 seconds at 30fps
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        random_frames.append(frame)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Add transition
    for _ in range(30):  # 1 second black
        random_frames.append(np.zeros_like(random_frames[0]))
    
    # Second: MuZero agent (5 seconds)
    print("2️⃣ Recording MuZero agent...")
    
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=20
    )
    
    if os.path.exists('muzero_breakout_best.pt'):
        muzero.load_checkpoint('muzero_breakout_best.pt')
        print("   Loaded trained model")
    
    muzero_frames = []
    obs, _ = env.reset()
    
    for _ in range(150):  # 5 seconds at 30fps
        action_probs = muzero.run_mcts(obs)
        action = np.argmax(action_probs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        muzero_frames.append(frame)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # Combine videos
    all_frames = random_frames + muzero_frames
    
    # Save comparison video
    video_path = 'breakout_comparison_10sec.mp4'
    with imageio.get_writer(video_path, fps=30) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"\n✅ Comparison video saved: {video_path}")
    print("   First 5 seconds: Random agent")
    print("   Last 5 seconds: MuZero agent")
    
    return video_path


if __name__ == "__main__":
    # Create main 10-second video
    video_path = create_breakout_10sec_video()
    
    # Optionally create comparison video
    # comparison_path = create_comparison_video()