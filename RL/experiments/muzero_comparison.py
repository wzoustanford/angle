#!/usr/bin/env python3
"""
Compare MuZero with existing RL algorithms on Atari games
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config.AgentConfig import AgentConfig
from config.MuZeroConfig import MuZeroConfig
from model.dqn_agent import DQNAgent
from model.muzero_agent import MuZeroAgent


def train_dqn(game: str, episodes: int, use_dueling: bool = False, 
              use_priority: bool = False) -> dict:
    """Train DQN agent and return results"""
    config = AgentConfig()
    config.env_name = f'ALE/{game}-v5'
    config.use_dueling = use_dueling
    config.use_prioritized_replay = use_priority
    
    agent = DQNAgent(config)
    
    print(f"\nTraining DQN (dueling={use_dueling}, priority={use_priority}) on {game}")
    
    rewards = []
    start_time = time.time()
    
    for episode in range(episodes):
        state, _ = agent.env.reset()
        state = agent.frame_stack.reset(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.push(next_state)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Training step
            if len(agent.replay_buffer) >= config.min_replay_size:
                agent.train_step()
        
        rewards.append(total_reward)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards[-5:])
            print(f"  Episode {episode + 1}/{episodes}: Avg Reward = {avg_reward:.1f}")
    
    training_time = time.time() - start_time
    
    return {
        'algorithm': f"DQN{'+Dueling' if use_dueling else ''}{'+Priority' if use_priority else ''}",
        'rewards': rewards,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'training_time': training_time
    }


def train_muzero(game: str, episodes: int) -> dict:
    """Train MuZero agent and return results"""
    config = MuZeroConfig()
    config.env_name = f'ALE/{game}-v5'
    config.num_simulations = 25  # Reduced for faster training in comparison
    config.batch_size = 64
    
    agent = MuZeroAgent(config)
    
    print(f"\nTraining MuZero on {game}")
    
    rewards = []
    start_time = time.time()
    
    # MuZero training is different - it uses self-play
    for episode in range(episodes):
        game_history = agent.self_play()
        total_reward = sum(game_history.rewards)
        rewards.append(total_reward)
        
        # Train after each game
        for _ in range(10):  # Multiple training steps per game
            agent.train_step()
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards[-5:])
            print(f"  Episode {episode + 1}/{episodes}: Avg Reward = {avg_reward:.1f}")
    
    training_time = time.time() - start_time
    
    return {
        'algorithm': 'MuZero',
        'rewards': rewards,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'training_time': training_time
    }


def plot_comparison(results: list, game: str, save_path: str):
    """Plot comparison of algorithms"""
    plt.figure(figsize=(12, 8))
    
    # Plot learning curves
    plt.subplot(2, 2, 1)
    for result in results:
        episodes = range(1, len(result['rewards']) + 1)
        plt.plot(episodes, result['rewards'], label=result['algorithm'], alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{game} - Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot moving average
    plt.subplot(2, 2, 2)
    window = 5
    for result in results:
        if len(result['rewards']) >= window:
            moving_avg = np.convolve(result['rewards'], 
                                    np.ones(window)/window, mode='valid')
            episodes = range(window, len(result['rewards']) + 1)
            plt.plot(episodes, moving_avg, label=result['algorithm'])
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (5-episode window)')
    plt.title(f'{game} - Moving Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bar chart of mean rewards
    plt.subplot(2, 2, 3)
    algorithms = [r['algorithm'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    
    bars = plt.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5)
    plt.ylabel('Mean Reward')
    plt.title(f'{game} - Mean Performance')
    plt.xticks(rotation=45, ha='right')
    
    # Color best performer
    best_idx = np.argmax(mean_rewards)
    bars[best_idx].set_color('green')
    
    # Training time comparison
    plt.subplot(2, 2, 4)
    training_times = [r['training_time'] for r in results]
    plt.bar(algorithms, training_times)
    plt.ylabel('Training Time (seconds)')
    plt.title(f'{game} - Training Time')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare MuZero with other RL algorithms')
    parser.add_argument('--game', type=str, default='SpaceInvaders',
                       help='Atari game to test on')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes to train')
    parser.add_argument('--output-dir', type=str, default='./results/muzero_comparison',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"=" * 60)
    print(f"MuZero vs Traditional RL Algorithms Comparison")
    print(f"Game: {args.game}")
    print(f"Episodes: {args.episodes}")
    print(f"=" * 60)
    
    results = []
    
    # Train different algorithms
    algorithms = [
        ('DQN', False, False),
        ('DQN+Dueling', True, False),
        ('DQN+Priority', False, True),
        ('MuZero', None, None)
    ]
    
    for alg_name, use_dueling, use_priority in algorithms:
        try:
            if alg_name == 'MuZero':
                result = train_muzero(args.game, args.episodes)
            else:
                result = train_dqn(args.game, args.episodes, use_dueling, use_priority)
            results.append(result)
        except Exception as e:
            print(f"Error training {alg_name}: {e}")
            continue
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {args.game}")
    print(f"{'=' * 60}")
    
    # Sort by mean reward
    results.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['algorithm']}")
        print(f"   Mean Reward: {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")
        print(f"   Max Reward: {result['max_reward']:.1f}")
        print(f"   Training Time: {result['training_time']:.1f}s")
    
    # Save results to JSON
    json_path = os.path.join(args.output_dir, f'results_{args.game}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Create comparison plot
    plot_path = os.path.join(args.output_dir, f'comparison_{args.game}_{timestamp}.png')
    plot_comparison(results, args.game, plot_path)
    
    # Highlight MuZero performance
    muzero_result = next((r for r in results if r['algorithm'] == 'MuZero'), None)
    if muzero_result:
        print(f"\n{'=' * 60}")
        print(f"MUZERO PERFORMANCE ANALYSIS")
        print(f"{'=' * 60}")
        
        # Compare to best DQN variant
        dqn_results = [r for r in results if 'DQN' in r['algorithm']]
        if dqn_results:
            best_dqn = max(dqn_results, key=lambda x: x['mean_reward'])
            improvement = ((muzero_result['mean_reward'] - best_dqn['mean_reward']) / 
                         abs(best_dqn['mean_reward']) * 100 if best_dqn['mean_reward'] != 0 else 0)
            
            print(f"MuZero vs Best DQN ({best_dqn['algorithm']}):")
            print(f"  Reward improvement: {improvement:+.1f}%")
            print(f"  MuZero: {muzero_result['mean_reward']:.1f}")
            print(f"  {best_dqn['algorithm']}: {best_dqn['mean_reward']:.1f}")


if __name__ == '__main__':
    main()