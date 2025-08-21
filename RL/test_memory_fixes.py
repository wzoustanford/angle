#!/usr/bin/env python3
"""
Test memory usage after fixing computation graph leaks.
Monitors memory usage over multiple episodes to detect leaks.
"""

import sys
import os
import gc
import torch
import psutil
import numpy as np
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent


def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process()
    mem_info = {
        'process_mb': process.memory_info().rss / 1024 / 1024,
        'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
    }
    if torch.cuda.is_available():
        mem_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        mem_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    return mem_info


def test_algorithm(name, config, num_episodes=10):
    """Test an algorithm and monitor memory usage"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Episodes: {num_episodes}")
    print("="*60)
    
    memory_history = []
    episode_rewards = []
    
    # Initial memory
    initial_mem = get_memory_info()
    memory_history.append(initial_mem)
    print(f"Initial memory: {initial_mem['process_mb']:.1f}MB")
    if 'gpu_allocated_mb' in initial_mem:
        print(f"Initial GPU: {initial_mem['gpu_allocated_mb']:.1f}MB allocated, "
              f"{initial_mem['gpu_reserved_mb']:.1f}MB reserved")
    
    try:
        agent = DQNAgent(config)
        
        for ep in range(num_episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            
            # Reset LSTM state for R2D2
            if hasattr(agent, 'reset_hidden_state'):
                agent.reset_hidden_state()
            
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 1000  # Moderate length episodes
            
            while not done and steps < max_steps:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                next_state = agent.frame_stack.append(next_obs)
                
                # Store transition
                if hasattr(agent.replay_buffer, 'push_transition'):
                    agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                else:
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update networks
                if agent.steps_done % 4 == 0 and len(agent.replay_buffer) > 100:
                    agent.update_q_network()
                
                if agent.steps_done % 500 == 0:
                    agent.update_target_network()
                
                state = next_state
                episode_reward += reward
                steps += 1
                agent.steps_done += 1
            
            agent.epsilon *= 0.995
            episode_rewards.append(episode_reward)
            
            # Check memory after each episode
            current_mem = get_memory_info()
            memory_history.append(current_mem)
            
            mem_increase = current_mem['process_mb'] - initial_mem['process_mb']
            print(f"Episode {ep+1:2d}: Reward={episode_reward:6.1f}, "
                  f"Memory={current_mem['process_mb']:7.1f}MB "
                  f"(+{mem_increase:6.1f}MB)")
            
            if 'gpu_allocated_mb' in current_mem:
                gpu_increase = current_mem['gpu_allocated_mb'] - initial_mem['gpu_allocated_mb']
                print(f"            GPU={current_mem['gpu_allocated_mb']:7.1f}MB allocated "
                      f"(+{gpu_increase:6.1f}MB)")
        
        # Final statistics
        print("\n" + "-"*60)
        print(f"Average reward: {np.mean(episode_rewards):.1f}")
        
        # Memory growth analysis
        final_mem = memory_history[-1]
        total_increase = final_mem['process_mb'] - initial_mem['process_mb']
        avg_increase_per_ep = total_increase / num_episodes
        
        print(f"\nMemory Growth Analysis:")
        print(f"  Total increase: {total_increase:.1f}MB")
        print(f"  Per episode: {avg_increase_per_ep:.1f}MB")
        
        if 'gpu_allocated_mb' in final_mem:
            gpu_total = final_mem['gpu_allocated_mb'] - initial_mem['gpu_allocated_mb']
            gpu_per_ep = gpu_total / num_episodes
            print(f"  GPU total increase: {gpu_total:.1f}MB")
            print(f"  GPU per episode: {gpu_per_ep:.1f}MB")
        
        # Check for memory leak
        if avg_increase_per_ep > 50:  # More than 50MB per episode is suspicious
            print("⚠️  WARNING: Possible memory leak detected!")
        else:
            print("✓ Memory usage appears stable")
        
        # Cleanup
        agent.env.close()
        del agent
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    time.sleep(2)
    
    # Final memory after cleanup
    final_cleanup_mem = get_memory_info()
    cleanup_amount = current_mem['process_mb'] - final_cleanup_mem['process_mb']
    print(f"\nAfter cleanup: {final_cleanup_mem['process_mb']:.1f}MB "
          f"(freed {cleanup_amount:.1f}MB)")
    
    return memory_history, episode_rewards


def main():
    """Run memory leak tests"""
    print("="*60)
    print("Memory Leak Test - After Fixes")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test 1: Standard DQN (baseline)
    config = AgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.memory_size = 5000
    config.batch_size = 32
    config.min_replay_size = 100
    test_algorithm("Standard DQN", config, num_episodes=10)
    
    # Test 2: R2D2 (LSTM - most likely to have memory leaks)
    config = AgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.use_r2d2 = True
    config.memory_size = 5000
    config.sequence_length = 40
    config.burn_in_length = 20
    config.lstm_size = 256
    config.min_replay_size = 100
    memory_history, rewards = test_algorithm("R2D2 (LSTM)", config, num_episodes=10)
    
    # Plot memory growth if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        episodes = list(range(len(memory_history)))
        process_mem = [m['process_mb'] for m in memory_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, process_mem, 'b-o', label='Process Memory')
        plt.xlabel('Episode')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage Over Episodes (After Fixes)')
        plt.grid(True)
        plt.legend()
        
        # Add trend line
        z = np.polyfit(episodes, process_mem, 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), "r--", alpha=0.5, 
                label=f'Trend: {z[0]:.1f}MB/episode')
        plt.legend()
        
        plt.savefig('memory_usage_after_fixes.png')
        print("\nMemory plot saved to memory_usage_after_fixes.png")
    except ImportError:
        print("\n(matplotlib not available for plotting)")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == '__main__':
    main()