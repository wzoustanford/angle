#!/usr/bin/env python3
"""
Alien and Ice Hockey Experiment
Tests RL algorithms on Alien and Ice Hockey games with 20 episodes each
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


class AlienIceHockeyExperiment:
    """
    Experiment for Alien and Ice Hockey games
    """
    
    def __init__(self, episodes=20, output_dir='./results/alien_icehockey'):
        self.episodes = episodes
        self.games = ['ALE/Alien-v5', 'ALE/IceHockey-v5']
        self.game_names = ['Alien', 'IceHockey']
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"experiment_{episodes}ep_{timestamp}")
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Alien and Ice Hockey Experiment")
        print(f"Episodes per algorithm per game: {self.episodes}")
        print(f"Output directory: {self.output_dir}")
    
    def create_basic_dqn_config(self, env_name: str) -> AgentConfig:
        """Basic DQN configuration"""
        config = AgentConfig()
        config.env_name = env_name
        config.use_r2d2 = False
        config.use_prioritized_replay = False
        config.use_dueling = False
        
        # Optimized for shorter training
        config.memory_size = 10000
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.min_replay_size = 500
        config.save_interval = 50000  # Disable frequent saving
        
        return config
    
    def create_dueling_dqn_config(self, env_name: str) -> AgentConfig:
        """DQN + Dueling Networks configuration"""
        config = self.create_basic_dqn_config(env_name)
        config.use_dueling = True  # Enable dueling architecture
        return config
    
    def create_prioritized_dqn_config(self, env_name: str) -> AgentConfig:
        """DQN + Prioritized Replay configuration"""
        config = self.create_basic_dqn_config(env_name)
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        return config
    
    def create_distributed_priority_config(self, env_name: str) -> DistributedAgentConfig:
        """Distributed RL with DQN + Priority configuration"""
        config = DistributedAgentConfig()
        config.env_name = env_name
        
        # Distributed settings
        config.num_workers = 4
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        config.use_dueling = False
        
        # Training parameters
        config.memory_size = 20000
        config.batch_size = 64
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.min_replay_size = 1000
        config.save_interval = 50000
        
        return config
    
    def run_algorithm(self, name: str, config, env_name: str) -> tuple:
        """Run a single algorithm and return results"""
        game_name = env_name.split('/')[-1].replace('-v5', '')
        print(f"\n{'='*60}")
        print(f"Running {name} on {game_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if isinstance(config, DistributedAgentConfig):
                # Distributed agent
                agent = DistributedDQNAgent(config, num_workers=config.num_workers)
                print(f"Starting distributed training with {config.num_workers} workers...")
                
                results = agent.train_distributed(total_episodes=self.episodes)
                
                # Extract episode rewards
                env_stats = results.get('env_stats', {})
                episode_rewards = []
                
                if 'worker_stats' in env_stats:
                    for worker_stat in env_stats['worker_stats']:
                        episodes = worker_stat.get('total_episodes', 0)
                        avg_reward = worker_stat.get('avg_reward', 0)
                        if episodes > 0 and not np.isnan(avg_reward):
                            episode_rewards.extend([avg_reward] * episodes)
                
                if not episode_rewards and 'overall_avg_reward' in env_stats:
                    overall_avg = env_stats['overall_avg_reward']
                    total_episodes = env_stats.get('total_episodes', 0)
                    if total_episodes > 0 and not np.isnan(overall_avg):
                        episode_rewards = [overall_avg] * total_episodes
                
                losses = results.get('training_stats', {}).get('losses', [])
                
            else:
                # Single-threaded agent
                agent = DQNAgent(config)
                print(f"Starting {name} training...")
                
                episode_rewards = []
                losses = []
                
                for episode in range(self.episodes):
                    episode_start_time = time.time()
                    
                    # Reset environment
                    obs, _ = agent.env.reset()
                    state = agent.frame_stack.reset(obs)
                    agent.reset_hidden_state()
                    episode_reward = 0
                    episode_losses = []
                    episode_steps = 0
                    
                    done = False
                    while not done:
                        # Select and perform action
                        action = agent.select_action(state)
                        next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                        done = terminated or truncated
                        
                        # Clip rewards if configured
                        if getattr(agent.config, 'clip_rewards', False):
                            reward = np.clip(reward, -1.0, 1.0)
                        
                        # Stack frames
                        next_state = agent.frame_stack.append(next_obs)
                        
                        # Store transition
                        if getattr(agent.config, 'use_r2d2', False):
                            agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                        else:
                            agent.replay_buffer.push(state, action, reward, next_state, done)
                        
                        # Update state
                        state = next_state
                        episode_reward += reward
                        agent.steps_done += 1
                        episode_steps += 1
                        
                        # Update Q-network
                        if agent.steps_done % agent.config.policy_update_interval == 0:
                            loss = agent.update_q_network()
                            if loss is not None:
                                episode_losses.append(loss)
                        
                        # Update target network
                        if agent.steps_done % agent.config.target_update_freq == 0:
                            agent.update_target_network()
                    
                    # Update exploration parameters
                    agent.epsilon = max(agent.config.epsilon_end, agent.epsilon * agent.config.epsilon_decay)
                    
                    if agent.config.use_prioritized_replay:
                        progress = min(episode / self.episodes, 1.0)
                        agent.priority_beta = agent.config.priority_beta_start + progress * (
                            agent.priority_beta_end - agent.config.priority_beta_start)
                        if hasattr(agent.replay_buffer, 'update_beta'):
                            agent.replay_buffer.update_beta(agent.priority_beta)
                    
                    # Record statistics
                    episode_rewards.append(episode_reward)
                    if episode_losses:
                        losses.append(np.mean(episode_losses))
                    
                    episode_time = time.time() - episode_start_time
                    print(f"  Episode {episode+1}/{self.episodes}: {episode_steps} steps, "
                          f"reward: {episode_reward:.1f}, time: {episode_time:.1f}s")
            
            elapsed = time.time() - start_time
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                final_avg = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else avg_reward
                best_reward = max(episode_rewards)
                
                print(f"✓ {name} on {game_name} completed in {elapsed:.1f}s")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Final 5 episodes: {final_avg:.2f}")
                print(f"  Best episode: {best_reward:.2f}")
            else:
                print(f"✓ {name} on {game_name} completed (no rewards recorded)")
                episode_rewards = [0] * self.episodes
                losses = []
            
            return episode_rewards, losses, elapsed
            
        except Exception as e:
            print(f"✗ Error in {name} on {game_name}: {e}")
            import traceback
            traceback.print_exc()
            return [0] * self.episodes, [], 0
    
    def run_all_experiments(self):
        """Run all experiments"""
        algorithms = [
            ('Basic DQN', lambda env: self.create_basic_dqn_config(env)),
            ('DQN + Dueling', lambda env: self.create_dueling_dqn_config(env)),
            ('DQN + Priority', lambda env: self.create_prioritized_dqn_config(env)),
            ('Distributed + Priority', lambda env: self.create_distributed_priority_config(env)),
        ]
        
        print(f"\nStarting experiments on {', '.join(self.game_names)}")
        print(f"Algorithms to test: {len(algorithms)}")
        print("="*60)
        
        # Initialize results
        for game_name in self.game_names:
            self.results[game_name] = {}
        
        # Run experiments for each game
        for env_name, game_name in zip(self.games, self.game_names):
            print(f"\n{'#'*60}")
            print(f"GAME: {game_name}")
            print(f"{'#'*60}")
            
            for alg_name, config_fn in algorithms:
                config = config_fn(env_name)
                rewards, losses, elapsed = self.run_algorithm(alg_name, config, env_name)
                
                self.results[game_name][alg_name] = {
                    'rewards': rewards,
                    'losses': losses,
                    'elapsed': elapsed
                }
                
                # Save intermediate results
                self.save_results()
    
    def create_comparison_plots(self):
        """Create comparison plots"""
        print("\nCreating comparison plots...")
        
        colors = {
            'Basic DQN': '#1f77b4',
            'DQN + Dueling': '#ff7f0e',
            'DQN + Priority': '#2ca02c',
            'Distributed + Priority': '#d62728'
        }
        
        # Create plots for both games
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Alien and Ice Hockey RL Comparison ({self.episodes} episodes)', 
                     fontsize=16, fontweight='bold')
        
        for i, game_name in enumerate(self.game_names):
            # Raw rewards plot
            ax1 = axes[i, 0]
            for alg_name, color in colors.items():
                if alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    if rewards:
                        episodes = range(1, len(rewards) + 1)
                        ax1.plot(episodes, rewards, color=color, alpha=0.7, linewidth=1, label=alg_name)
            
            ax1.set_title(f'{game_name} - Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Moving average plot
            ax2 = axes[i, 1]
            window = min(5, self.episodes // 4)
            
            for alg_name, color in colors.items():
                if alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    if len(rewards) >= window:
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        episodes = range(window, len(rewards) + 1)
                        ax2.plot(episodes, moving_avg, color=color, linewidth=2, label=alg_name)
            
            ax2.set_title(f'{game_name} - Moving Average ({window} episodes)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'alien_icehockey_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print results summary"""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        for game_name in self.game_names:
            print(f"\n{game_name}:")
            print("-" * 40)
            
            results_summary = []
            for alg_name in self.results[game_name].keys():
                rewards = self.results[game_name][alg_name]['rewards']
                if rewards:
                    avg_reward = np.mean(rewards)
                    final_avg = np.mean(rewards[-5:]) if len(rewards) >= 5 else avg_reward
                    max_reward = max(rewards)
                    elapsed = self.results[game_name][alg_name]['elapsed']
                    
                    results_summary.append((alg_name, avg_reward, final_avg, max_reward, elapsed))
            
            # Sort by final average
            results_summary.sort(key=lambda x: x[2], reverse=True)
            
            print(f"{'Algorithm':<20} {'Avg Reward':<12} {'Final 5':<12} {'Best':<12} {'Time':<8}")
            print("-" * 70)
            for i, (alg, avg, final, best, elapsed) in enumerate(results_summary):
                print(f"#{i+1} {alg:<18} {avg:>8.2f}     {final:>8.2f}     {best:>8.2f}     {elapsed:>6.1f}s")
    
    def save_results(self):
        """Save results to JSON"""
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f'results_{timestamp}.json')
        
        # Convert to JSON-serializable format
        json_results = {}
        for game in self.results:
            json_results[game] = {}
            for algorithm in self.results[game]:
                json_results[game][algorithm] = {
                    'rewards': [float(x) for x in self.results[game][algorithm]['rewards']],
                    'losses': [float(x) for x in self.results[game][algorithm]['losses']],
                    'elapsed': float(self.results[game][algorithm]['elapsed'])
                }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Alien and Ice Hockey RL Experiment')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Episodes per algorithm per game (default: 20)')
    parser.add_argument('--output-dir', type=str, default='./results/alien_icehockey',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        experiment = AlienIceHockeyExperiment(
            episodes=args.episodes,
            output_dir=args.output_dir
        )
        
        experiment.run_all_experiments()
        experiment.create_comparison_plots()
        experiment.print_summary()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED!")
        print(f"Results saved in: {experiment.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()