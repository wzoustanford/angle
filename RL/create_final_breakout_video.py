#!/usr/bin/env python3
"""
Create a 10-second video with muzero_breakout_best.pt
Named: muzero_breakout_10sec_best_before_finish.mp4
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


def create_final_video():
    """Create exactly 10-second video of MuZero playing Breakout with best model"""
    
    checkpoint_path = 'muzero_breakout_best.pt'
    video_path = 'muzero_breakout_10sec_best_before_finish.mp4'
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("Please ensure muzero_breakout_best.pt exists in the current directory")
        return None
    
    print("="*60)
    print("Creating Final Breakout Video")
    print("="*60)
    print(f"Model: {checkpoint_path}")
    print(f"Output: {video_path}")
    print()
    
    # Create environment
    env = AtariWrapper('ALE/Breakout-v5', frame_stack=4)
    
    # Initialize MuZero and load the best model
    print("Loading MuZero model...")
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=50,  # Good balance of quality and speed
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # Load the best checkpoint
    muzero.load_checkpoint(checkpoint_path)
    print(f"✅ Model loaded successfully")
    print(f"   Device: {muzero.device}")
    
    # Target: 10 seconds at 30 fps = 300 frames
    target_frames = 300
    fps = 30
    
    print(f"\n📹 Recording gameplay...")
    print(f"   Target: {target_frames} frames at {fps} fps (10 seconds)")
    
    # Initialize recording
    all_frames = []
    episode_count = 0
    total_score = 0
    episode_scores = []
    bricks_broken = 0
    
    # Start first episode
    obs, info = env.reset()
    current_lives = info.get('lives', 5)
    episode_count = 1
    episode_score = 0
    
    print(f"\nEpisode 1 started - Lives: {current_lives}")
    
    # Capture initial frame
    frame = env.render()
    all_frames.append(frame)
    
    while len(all_frames) < target_frames:
        # Use MCTS to select action
        action_probs = muzero.run_mcts(obs)
        
        # Greedy action selection for best performance
        action = np.argmax(action_probs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Capture frame
        frame = env.render()
        all_frames.append(frame)
        
        # Track score
        if reward > 0:
            bricks_broken += 1
            episode_score += reward
            total_score += reward
        
        # Check lives
        new_lives = info.get('lives', 0)
        if new_lives < current_lives:
            print(f"  Lost a life! Lives remaining: {new_lives}")
            current_lives = new_lives
        
        # If game ended and we need more frames, start a new episode
        if done and len(all_frames) < target_frames - 10:
            episode_scores.append(episode_score)
            print(f"  Episode {episode_count} ended - Score: {episode_score}")
            
            # Add transition frames (black)
            for _ in range(5):
                if len(all_frames) < target_frames:
                    all_frames.append(np.zeros_like(frame))
            
            # Reset for new episode
            episode_count += 1
            obs, info = env.reset()
            current_lives = info.get('lives', 5)
            episode_score = 0
            done = False
            print(f"\nEpisode {episode_count} started - Lives: {current_lives}")
    
    # Add final episode score if not done
    if not done:
        episode_scores.append(episode_score)
        print(f"  Episode {episode_count} ended - Score: {episode_score}")
    
    env.close()
    
    # Trim to exactly target frames
    all_frames = all_frames[:target_frames]
    
    # Save video
    print(f"\n💾 Saving video...")
    
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    # Calculate statistics
    duration = len(all_frames) / fps
    avg_score = np.mean(episode_scores) if episode_scores else 0
    
    print("\n" + "="*60)
    print("✅ Video Created Successfully!")
    print("="*60)
    print(f"📹 File: {video_path}")
    print(f"📊 Statistics:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Frames: {len(all_frames)}")
    print(f"   Episodes: {episode_count}")
    print(f"   Total Score: {total_score}")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Bricks Broken: {bricks_broken}")
    print(f"   Episode Scores: {episode_scores}")
    print("\n🎬 To play the video:")
    print(f"   ffplay {video_path}")
    print("="*60)
    
    return video_path


if __name__ == "__main__":
    video_path = create_final_video()
    
    if video_path:
        # Also check file size
        import os
        file_size = os.path.getsize(video_path) / 1024  # KB
        print(f"\n📁 File size: {file_size:.1f} KB")