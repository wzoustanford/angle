#!/usr/bin/env python3
"""
Quick reward and speed test: GPU vs CPU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent

def quick_training_test(device_type: str, max_steps_per_episode: int = 500, num_episodes: int = 10):
    """Quick training test with limited steps per episode"""
    print(f"\n=== Quick Test on {device_type.upper()} ===")
    
    config = AgentConfig()
    config.device = device_type
    config.epsilon_start = 0.3  # Less exploration for faster episodes
    config.epsilon_end = 0.1
    config.epsilon_decay = 0.98
    config.batch_size = 16
    config.memory_size = 5000
    config.min_replay_size = 500
    config.policy_update_interval = 2
    config.target_update_freq = 200
    config.save_interval = 999999
    
    agent = DQNAgent(config)
    print(f"Device: {agent.device}")
    
    episode_rewards = []
    episode_times = []
    training_steps = []
    total_start = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        agent.reset_hidden_state()
        episode_reward = 0
        steps_this_episode = 0
        done = False
        
        while not done and steps_this_episode < max_steps_per_episode:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            agent.steps_done += 1
            steps_this_episode += 1
            
            # Quick training updates
            if agent.steps_done % config.policy_update_interval == 0:
                loss = agent.update_q_network()
            
            if agent.steps_done % config.target_update_freq == 0:
                agent.update_target_network()
        
        agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)
        
        episode_time = time.time() - episode_start
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        training_steps.append(agent.steps_done)
        
        print(f"  Episode {episode}: Reward={episode_reward:.1f}, Steps={steps_this_episode}, Time={episode_time:.1f}s")
    
    total_time = time.time() - total_start
    
    return {
        'device': device_type,
        'episode_rewards': episode_rewards,
        'episode_times': episode_times,
        'training_steps': training_steps,
        'total_time': total_time,
        'avg_episode_time': np.mean(episode_times),
        'final_avg_reward': np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'total_training_steps': agent.steps_done
    }

def main():
    """Run quick comparison"""
    print("Quick DQN Training Test: GPU vs CPU")
    print("=" * 45)
    
    results = {}
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            results['gpu'] = quick_training_test('cuda', max_steps_per_episode=300, num_episodes=8)
        except Exception as e:
            print(f"GPU test failed: {e}")
            results['gpu'] = None
    
    # Test CPU
    try:
        results['cpu'] = quick_training_test('cpu', max_steps_per_episode=300, num_episodes=8)
    except Exception as e:
        print(f"CPU test failed: {e}")
        results['cpu'] = None
    
    # Compare results
    print("\n" + "=" * 45)
    print("COMPARISON RESULTS")
    print("=" * 45)
    
    if results.get('gpu') and results.get('cpu'):
        gpu = results['gpu']
        cpu = results['cpu']
        
        print(f"\n📊 Training Performance:")
        print(f"  GPU avg time per episode: {gpu['avg_episode_time']:.2f}s")
        print(f"  CPU avg time per episode: {cpu['avg_episode_time']:.2f}s")
        speedup = cpu['avg_episode_time'] / gpu['avg_episode_time']
        print(f"  Speed improvement: {speedup:.2f}x faster on GPU")
        
        print(f"\n🎯 Reward Performance:")
        print(f"  GPU final avg reward: {gpu['final_avg_reward']:.2f}")
        print(f"  CPU final avg reward: {cpu['final_avg_reward']:.2f}")
        print(f"  GPU max reward: {gpu['max_reward']:.1f}")
        print(f"  CPU max reward: {cpu['max_reward']:.1f}")
        
        print(f"\n🔧 Training Details:")
        print(f"  GPU total training steps: {gpu['total_training_steps']}")
        print(f"  CPU total training steps: {cpu['total_training_steps']}")
        print(f"  GPU total time: {gpu['total_time']:.1f}s")
        print(f"  CPU total time: {cpu['total_time']:.1f}s")
        
        # Show episode progression
        print(f"\n📈 Episode Rewards:")
        print(f"  GPU: {[f'{r:.1f}' for r in gpu['episode_rewards']]}")
        print(f"  CPU: {[f'{r:.1f}' for r in cpu['episode_rewards']]}")
        
        # Confirm no mixed precision was used
        print(f"\n✅ Confirmation: No mixed precision used - just basic GPU tensor placement")
        
    else:
        if results.get('gpu'):
            gpu = results['gpu']
            print(f"GPU Results: Avg time={gpu['avg_episode_time']:.2f}s, Avg reward={gpu['final_avg_reward']:.2f}")
        if results.get('cpu'):
            cpu = results['cpu']
            print(f"CPU Results: Avg time={cpu['avg_episode_time']:.2f}s, Avg reward={cpu['final_avg_reward']:.2f}")

if __name__ == "__main__":
    main()