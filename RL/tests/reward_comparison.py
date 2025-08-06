#!/usr/bin/env python3
"""
Compare training reward curves: GPU vs CPU for Space Invaders DQN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent

def train_agent_with_tracking(device_type: str, num_episodes: int = 50):
    """Train agent and track rewards over time"""
    print(f"\n=== Training on {device_type.upper()} ===")
    
    # Configure for reward tracking
    config = AgentConfig()
    config.device = device_type
    config.epsilon_start = 1.0
    config.epsilon_end = 0.1
    config.epsilon_decay = 0.995
    config.batch_size = 32
    config.memory_size = 10000
    config.min_replay_size = 1000
    config.policy_update_interval = 4
    config.target_update_freq = 1000
    config.save_interval = 999999  # Disable saving
    config.learning_rate = 1e-4
    
    # Create agent
    agent = DQNAgent(config)
    print(f"Using device: {agent.device}")
    
    # Track training progress
    episode_rewards = []
    episode_times = []
    training_start = time.time()
    
    # Train with progress tracking
    for episode in range(num_episodes):
        episode_start = time.time()
        
        # Run single episode
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        agent.reset_hidden_state()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            
            # Store transition
            if getattr(config, 'use_r2d2', False):
                agent.replay_buffer.push_transition(state, action, reward, next_state, done)
            else:
                agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            agent.steps_done += 1
            
            # Update networks
            if agent.steps_done % config.policy_update_interval == 0:
                loss = agent.update_q_network()
            
            if agent.steps_done % config.target_update_freq == 0:
                agent.update_target_network()
        
        # Update epsilon
        agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)
        
        episode_time = time.time() - episode_start
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Reward={episode_reward:.1f}, Avg={avg_reward:.1f}, Time={episode_time:.1f}s")
    
    total_time = time.time() - training_start
    
    return {
        'device': device_type,
        'episode_rewards': episode_rewards,
        'episode_times': episode_times,
        'total_time': total_time,
        'final_avg_reward': np.mean(episode_rewards[-10:]),
        'total_episodes': len(episode_rewards)
    }

def plot_reward_curves(results):
    """Plot reward curves for comparison"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Raw rewards
    plt.subplot(1, 3, 1)
    for device, data in results.items():
        if data:
            episodes = range(len(data['episode_rewards']))
            plt.plot(episodes, data['episode_rewards'], label=f"{device.upper()}", alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Raw Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    plt.subplot(1, 3, 2)
    for device, data in results.items():
        if data:
            rewards = data['episode_rewards']
            # 10-episode moving average
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - 9)
                moving_avg.append(np.mean(rewards[start_idx:i+1]))
            
            episodes = range(len(moving_avg))
            plt.plot(episodes, moving_avg, label=f"{device.upper()}", linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('10-Episode Moving Average')
    plt.title('Smoothed Reward Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Episode times
    plt.subplot(1, 3, 3)
    for device, data in results.items():
        if data:
            episodes = range(len(data['episode_times']))
            plt.plot(episodes, data['episode_times'], label=f"{device.upper()}", alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Time (seconds)')
    plt.title('Episode Training Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/code/angle/RL/results/reward_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Reward curves saved to results/reward_comparison.png")
    
    return plt

def compare_training():
    """Compare GPU vs CPU training performance and rewards"""
    print("DQN Training Comparison: GPU vs CPU")
    print("=" * 50)
    
    # Ensure results directory exists
    os.makedirs('/home/ubuntu/code/angle/RL/results', exist_ok=True)
    
    num_episodes = 30  # Moderate number for meaningful comparison
    results = {}
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            print("Training on GPU...")
            results['gpu'] = train_agent_with_tracking('cuda', num_episodes)
        except Exception as e:
            print(f"GPU training failed: {e}")
            results['gpu'] = None
    else:
        print("CUDA not available")
        results['gpu'] = None
    
    # Test CPU
    try:
        print("Training on CPU...")
        results['cpu'] = train_agent_with_tracking('cpu', num_episodes)
    except Exception as e:
        print(f"CPU training failed: {e}")
        results['cpu'] = None
    
    # Analysis and plotting
    print("\n" + "=" * 50)
    print("TRAINING COMPARISON RESULTS")
    print("=" * 50)
    
    if results['gpu'] and results['cpu']:
        gpu_data = results['gpu']
        cpu_data = results['cpu']
        
        print(f"\nFinal Performance (last 10 episodes average):")
        print(f"  GPU: {gpu_data['final_avg_reward']:.2f}")
        print(f"  CPU: {cpu_data['final_avg_reward']:.2f}")
        
        print(f"\nTraining Speed:")
        gpu_time_per_ep = gpu_data['total_time'] / gpu_data['total_episodes']
        cpu_time_per_ep = cpu_data['total_time'] / cpu_data['total_episodes']
        speedup = cpu_time_per_ep / gpu_time_per_ep
        
        print(f"  GPU: {gpu_time_per_ep:.2f}s per episode")
        print(f"  CPU: {cpu_time_per_ep:.2f}s per episode")
        print(f"  Speedup: {speedup:.2f}x faster on GPU")
        
        print(f"\nTotal Training Time:")
        print(f"  GPU: {gpu_data['total_time']:.1f}s")
        print(f"  CPU: {cpu_data['total_time']:.1f}s")
        
        # Plot results
        plot_reward_curves(results)
        
        # Statistical comparison
        gpu_rewards = gpu_data['episode_rewards']
        cpu_rewards = cpu_data['episode_rewards']
        
        print(f"\nReward Statistics:")
        print(f"  GPU - Mean: {np.mean(gpu_rewards):.2f}, Std: {np.std(gpu_rewards):.2f}")
        print(f"  CPU - Mean: {np.mean(cpu_rewards):.2f}, Std: {np.std(cpu_rewards):.2f}")
        print(f"  Max rewards - GPU: {np.max(gpu_rewards):.1f}, CPU: {np.max(cpu_rewards):.1f}")
        
    else:
        print("Could not complete both GPU and CPU training for comparison")
    
    return results

if __name__ == "__main__":
    compare_training()