#!/usr/bin/env python3
"""
Quick train and play MuZero on CartPole with video recording
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import warnings
warnings.filterwarnings('ignore')

def quick_train_and_play():
    """Quick training and video recording"""
    
    print("Quick MuZero training on CartPole...")
    
    # Create environment
    env = gymnasium.make('CartPole-v1')
    
    # Initialize MuZero with settings for faster convergence
    muzero = SimpleMuZero(
        observation_shape=env.observation_space.shape,
        action_space_size=env.action_space.n,
        num_simulations=20,  # Fewer simulations for speed
        batch_size=32,
        max_moves=500,
        lr=1e-3,
        td_steps=5,
        num_unroll_steps=3
    )
    
    print(f"Device: {muzero.device}\n")
    
    # Quick training - 50 episodes
    print("Training for 50 episodes...")
    best_reward = 0
    rewards = []
    
    for episode in range(50):
        trajectory = muzero.self_play_game(env)
        muzero.update_replay_buffer(trajectory)
        
        episode_reward = sum(exp.reward for exp in trajectory)
        rewards.append(episode_reward)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            muzero.save_checkpoint('muzero_quick_best.pt')
        
        # Train after initial collection
        if episode >= 5:
            for _ in range(20):
                muzero.train_step()
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode+1}: Avg={avg_reward:.1f}, Best={best_reward:.0f}")
    
    print(f"\n✅ Training complete! Best reward: {best_reward}")
    
    # Now record video with the best model
    print("\n📹 Recording gameplay video...")
    
    # Load best model
    muzero.load_checkpoint('muzero_quick_best.pt')
    
    # Create environment for recording
    env_record = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    
    # Play 3 episodes
    all_frames = []
    play_rewards = []
    
    for ep in range(3):
        obs, _ = env_record.reset()
        frames = [env_record.render()]
        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            # Use MCTS for action selection
            action_probs = muzero.run_mcts(obs)
            # Greedy action selection for evaluation
            action = np.argmax(action_probs)
            
            obs, reward, terminated, truncated, _ = env_record.step(action)
            done = terminated or truncated
            
            frames.append(env_record.render())
            ep_reward += reward
            steps += 1
        
        play_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: {ep_reward:.0f} reward, {steps} steps")
        
        # Add separator frames between episodes
        if ep < 2:
            for _ in range(10):
                frames.append(np.zeros_like(frames[0]))
        
        all_frames.extend(frames)
    
    env_record.close()
    
    # Save video
    video_path = 'muzero_cartpole_demo.mp4'
    with imageio.get_writer(video_path, fps=30) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"\n✅ Video saved to {video_path}")
    print(f"Average performance: {np.mean(play_rewards):.1f} reward")
    print(f"Video duration: {len(all_frames)/30:.1f} seconds")
    
    # Also create a comparison with random agent
    print("\n📹 Creating random agent comparison...")
    
    env_random = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    random_frames = []
    
    for ep in range(2):
        obs, _ = env_random.reset()
        frames = [env_random.render()]
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env_random.action_space.sample()  # Random action
            obs, _, terminated, truncated, _ = env_random.step(action)
            done = terminated or truncated
            frames.append(env_random.render())
            steps += 1
        
        print(f"  Random Episode {ep+1}: {steps} steps")
        random_frames.extend(frames)
    
    env_random.close()
    
    # Combine: Random agent first, then trained agent
    comparison_frames = random_frames
    # Add transition frames
    for _ in range(20):
        comparison_frames.append(np.ones_like(random_frames[0]) * 128)  # Gray frames
    comparison_frames.extend(all_frames)
    
    comparison_path = 'muzero_vs_random.mp4'
    with imageio.get_writer(comparison_path, fps=30) as writer:
        for frame in comparison_frames:
            writer.append_data(frame)
    
    print(f"\n✅ Comparison video saved to {comparison_path}")
    print("  First part: Random agent")
    print("  Second part: Trained MuZero agent")
    
    env.close()

if __name__ == "__main__":
    quick_train_and_play()