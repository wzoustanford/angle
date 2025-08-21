#!/usr/bin/env python3
"""
Debug video generation with frame skipping
Check if we're capturing frames correctly
"""

import gymnasium
import ale_py
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import cv2
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AtariWrapperDebug:
    """
    Debug version that shows what's happening with frames
    """
    def __init__(self, env_name='ALE/Breakout-v5', frame_stack=4):
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        self.observation_shape = (frame_stack, 84, 84)
        self.step_count = 0
        
    def reset(self):
        obs, info = self.env.reset()
        obs, _, _, _, info = self.env.step(1)  # FIRE
        self._last_info = info
        
        processed = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        print(f"RESET: Lives={info.get('lives', 5)}")
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """
        The problem: We take 4 env steps but only capture 1 frame!
        """
        total_reward = 0
        done = False
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        
        # Store ALL frames during frame skip
        skipped_frames = []
        
        print(f"Step {self.step_count}: Action={action} ", end="")
        
        for skip in range(4):  # Frame skip
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Capture EACH frame during skip
            frame = self.env.render()
            skipped_frames.append(frame)
            
            if done:
                break
        
        self.step_count += 1
        
        # Check if lost life
        lives_after = info.get('lives', 0)
        if lives_before > 0 and lives_after < lives_before and lives_after > 0:
            obs, _, _, _, _ = self.env.step(1)  # FIRE
            print(f"-> Lost life! FIRED. Lives={lives_after}", end="")
        
        self._last_info = info
        
        # Only add final frame to state
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        print(f" -> Reward={total_reward:.0f}, Frames skipped={len(skipped_frames)}")
        
        # Return the skipped frames for video recording
        return self._get_stacked_frames(), total_reward, terminated, truncated, info, skipped_frames
    
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


def test_video_with_proper_frames():
    """
    Create video capturing ALL frames, not just one per step
    """
    print("="*60)
    print("DEBUG: Video Generation with Frame Skipping")
    print("="*60)
    
    env = AtariWrapperDebug('ALE/Breakout-v5')
    
    # Simple test: random actions
    print("\nTesting 100 environment steps (400 actual frames with skip=4)")
    print("-"*60)
    
    all_frames = []
    obs, info = env.reset()
    
    # Capture initial frame
    initial_frame = env.render()
    all_frames.append(initial_frame)
    
    total_reward = 0
    for step in range(100):  # 100 agent decisions
        # Random action
        action = np.random.choice([0, 2, 3])  # NOOP, LEFT, RIGHT
        
        # Step with frame skip
        obs, reward, terminated, truncated, info, skipped_frames = env.step(action)
        
        # ADD ALL SKIPPED FRAMES to video!
        all_frames.extend(skipped_frames)
        
        total_reward += reward
        
        if terminated or truncated:
            print(f"\nGame ended at step {step}")
            break
    
    env.close()
    
    print(f"\nTotal frames captured: {len(all_frames)}")
    print(f"Expected ~400 frames (100 steps × 4 skips)")
    print(f"Total reward: {total_reward}")
    
    # Save debug video
    video_path = 'debug_frame_skip_video.mp4'
    with imageio.get_writer(video_path, fps=60) as writer:  # 60 fps for actual game speed
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"\n✅ Debug video saved: {video_path}")
    print(f"   Duration at 60fps: {len(all_frames)/60:.1f} seconds")
    print(f"   Duration at 30fps: {len(all_frames)/30:.1f} seconds")
    
    return all_frames


def create_comparison_videos():
    """
    Create two videos showing the difference
    """
    print("\n" + "="*60)
    print("Creating Comparison Videos")
    print("="*60)
    
    # Video 1: One frame per step (WRONG - choppy)
    print("\n1. Creating WRONG video (1 frame per decision)...")
    env = gymnasium.make('ALE/Breakout-v5', render_mode='rgb_array')
    obs, _ = env.reset()
    env.step(1)  # FIRE
    
    frames_wrong = []
    for _ in range(50):
        frame = env.render()
        frames_wrong.append(frame)
        
        action = np.random.choice([0, 2, 3])
        # Skip 4 frames but only capture final one
        for _ in range(4):
            obs, r, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    
    with imageio.get_writer('breakout_wrong_choppy.mp4', fps=15) as writer:
        for frame in frames_wrong:
            writer.append_data(frame)
    print(f"   Saved: breakout_wrong_choppy.mp4 ({len(frames_wrong)} frames)")
    
    env.close()
    
    # Video 2: All frames (CORRECT - smooth)
    print("\n2. Creating CORRECT video (all frames)...")
    env = gymnasium.make('ALE/Breakout-v5', render_mode='rgb_array')
    obs, _ = env.reset()
    env.step(1)  # FIRE
    
    frames_correct = []
    for _ in range(50):
        action = np.random.choice([0, 2, 3])
        # Capture ALL frames during skip
        for _ in range(4):
            frame = env.render()
            frames_correct.append(frame)
            obs, r, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    
    with imageio.get_writer('breakout_correct_smooth.mp4', fps=60) as writer:
        for frame in frames_correct:
            writer.append_data(frame)
    print(f"   Saved: breakout_correct_smooth.mp4 ({len(frames_correct)} frames)")
    
    env.close()
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("\nThe PROBLEM with current video generation:")
    print("  ❌ We only capture 1 frame per agent decision")
    print("  ❌ With frame_skip=4, we miss 75% of the actual gameplay!")
    print("  ❌ Video looks choppy and ball teleports")
    print("\nThe SOLUTION:")
    print("  ✅ Capture ALL frames during frame skipping")
    print("  ✅ Save at 60fps (game's native framerate)")
    print("  ✅ Or capture every frame in a separate render loop")


if __name__ == "__main__":
    # Test the frame capture
    frames = test_video_with_proper_frames()
    
    # Create comparison videos
    create_comparison_videos()