#!/usr/bin/env python3
"""
Simplified test version of Alien and Ice Hockey experiment
Tests only Basic DQN with 2 episodes for quick validation
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model import DQNAgent


def run_quick_test():
    """Run quick test with 2 episodes per game"""
    games = ['ALE/Alien-v5', 'ALE/IceHockey-v5']
    game_names = ['Alien', 'IceHockey']
    episodes = 2
    max_steps_per_episode = 1000  # Limit steps to prevent hanging
    
    print("="*60)
    print("QUICK TEST: Alien and Ice Hockey")
    print(f"Running {episodes} episodes per game")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*60)
    
    results = {}
    
    for env_name, game_name in zip(games, game_names):
        print(f"\n{'#'*60}")
        print(f"GAME: {game_name}")
        print(f"{'#'*60}")
        
        # Create config
        config = AgentConfig()
        config.env_name = env_name
        config.use_r2d2 = False
        config.use_prioritized_replay = False
        config.memory_size = 5000
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.min_replay_size = 500
        config.save_interval = 50000
        
        try:
            # Create agent
            print(f"\nCreating DQN agent for {game_name}...")
            agent = DQNAgent(config)
            
            episode_rewards = []
            
            # Run episodes
            for episode in range(episodes):
                start_time = time.time()
                print(f"\nEpisode {episode+1}/{episodes}:")
                
                # Reset environment
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                agent.reset_hidden_state()
                episode_reward = 0
                steps = 0
                
                done = False
                while not done and steps < max_steps_per_episode:
                    # Select and perform action
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                    done = terminated or truncated
                    
                    # Stack frames
                    next_state = agent.frame_stack.append(next_obs)
                    
                    # Store transition
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    agent.steps_done += 1
                    steps += 1
                    
                    # Update Q-network periodically
                    if agent.steps_done % agent.config.policy_update_interval == 0:
                        agent.update_q_network()
                    
                    # Update target network
                    if agent.steps_done % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                    
                    # Progress update
                    if steps % 200 == 0:
                        print(f"  Step {steps}/{max_steps_per_episode}, reward: {episode_reward:.1f}")
                
                # Update exploration
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                episode_time = time.time() - start_time
                episode_rewards.append(episode_reward)
                print(f"  Completed: {steps} steps, reward: {episode_reward:.1f}, time: {episode_time:.1f}s")
            
            # Store results
            results[game_name] = {
                'rewards': episode_rewards,
                'avg_reward': np.mean(episode_rewards),
                'max_reward': max(episode_rewards),
                'min_reward': min(episode_rewards)
            }
            
            print(f"\n✓ {game_name} completed successfully")
            print(f"  Average reward: {results[game_name]['avg_reward']:.2f}")
            print(f"  Rewards: {episode_rewards}")
            
        except Exception as e:
            print(f"\n✗ Error with {game_name}: {e}")
            import traceback
            traceback.print_exc()
            results[game_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("="*60)
    
    for game_name in game_names:
        if game_name in results:
            if 'error' in results[game_name]:
                print(f"{game_name}: ERROR - {results[game_name]['error']}")
            else:
                print(f"{game_name}:")
                print(f"  Episodes run: {episodes}")
                print(f"  Average reward: {results[game_name]['avg_reward']:.2f}")
                print(f"  Max reward: {results[game_name]['max_reward']:.2f}")
                print(f"  Min reward: {results[game_name]['min_reward']:.2f}")
    
    print(f"\n✓ Quick test completed!")
    return results


if __name__ == '__main__':
    results = run_quick_test()