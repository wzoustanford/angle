#!/usr/bin/env python3
"""
Quick full test with 3 episodes per algorithm per game for faster results
Includes comprehensive logging and results generation
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


class QuickFullExperiment:
    """Quick experiment with comprehensive logging"""
    
    def __init__(self, episodes=3, output_dir='./experiments/results'):
        self.episodes = episodes
        self.games = ['ALE/Alien-v5', 'ALE/IceHockey-v5']
        self.game_names = ['Alien', 'IceHockey']
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"quick_experiment_{episodes}ep_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.start_time = time.time()
        
        # Create log file
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log(f"Quick Full Experiment Started")
        self.log(f"Episodes per algorithm per game: {episodes}")
        self.log(f"Games: {', '.join(self.game_names)}")
        self.log(f"Output directory: {self.output_dir}")
        self.log("="*60)
    
    def log(self, message):
        """Write to both console and log file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def create_basic_dqn_config(self, env_name: str) -> AgentConfig:
        """Basic DQN configuration"""
        config = AgentConfig()
        config.env_name = env_name
        config.use_dueling = False
        config.use_prioritized_replay = False
        config.memory_size = 5000
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.min_replay_size = 500
        config.save_interval = 50000
        return config
    
    def create_dueling_dqn_config(self, env_name: str) -> AgentConfig:
        """DQN + Dueling Networks configuration"""
        config = self.create_basic_dqn_config(env_name)
        config.use_dueling = True
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
    
    def run_single_threaded_algorithm(self, name: str, config, game_name: str) -> tuple:
        """Run single-threaded algorithm"""
        self.log(f"\n  Running {name} on {game_name}")
        start_time = time.time()
        max_steps_per_episode = 1500  # Reasonable limit
        
        try:
            agent = DQNAgent(config)
            episode_rewards = []
            losses = []
            
            for episode in range(self.episodes):
                episode_start = time.time()
                
                # Reset environment
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                agent.reset_hidden_state()
                episode_reward = 0
                episode_losses = []
                steps = 0
                
                done = False
                while not done and steps < max_steps_per_episode:
                    # Select action
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                    done = terminated or truncated
                    
                    # Process transition
                    next_state = agent.frame_stack.append(next_obs)
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Update
                    state = next_state
                    episode_reward += reward
                    agent.steps_done += 1
                    steps += 1
                    
                    # Train networks
                    if agent.steps_done % agent.config.policy_update_interval == 0:
                        loss = agent.update_q_network()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    if agent.steps_done % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                
                # Update exploration
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                if agent.config.use_prioritized_replay:
                    progress = min(episode / self.episodes, 1.0)
                    agent.priority_beta = agent.config.priority_beta_start + progress * (
                        agent.priority_beta_end - agent.config.priority_beta_start)
                    if hasattr(agent.replay_buffer, 'update_beta'):
                        agent.replay_buffer.update_beta(agent.priority_beta)
                
                episode_rewards.append(episode_reward)
                if episode_losses:
                    losses.append(np.mean(episode_losses))
                
                episode_time = time.time() - episode_start
                self.log(f"    Episode {episode+1}/{self.episodes}: {steps} steps, "
                        f"reward={episode_reward:.1f}, time={episode_time:.1f}s")
            
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards)
            self.log(f"  ✓ {name} completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
            
            return episode_rewards, losses, elapsed
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            return [0] * self.episodes, [], 0
    
    def run_all_experiments(self):
        """Run all experiments"""
        # Only test single-threaded algorithms for speed
        algorithms = [
            ('Basic DQN', lambda env: self.create_basic_dqn_config(env)),
            ('DQN + Dueling', lambda env: self.create_dueling_dqn_config(env)),
            ('DQN + Priority', lambda env: self.create_prioritized_dqn_config(env)),
        ]
        
        self.log(f"\nStarting experiments on {', '.join(self.game_names)}")
        self.log(f"Algorithms: {[name for name, _ in algorithms]}")
        
        # Initialize results
        for game_name in self.game_names:
            self.results[game_name] = {}
        
        # Run experiments
        total_experiments = len(self.games) * len(algorithms)
        current = 0
        
        for env_name, game_name in zip(self.games, self.game_names):
            self.log(f"\n{'='*40}")
            self.log(f"GAME: {game_name}")
            self.log(f"{'='*40}")
            
            for alg_name, config_fn in algorithms:
                current += 1
                self.log(f"\nExperiment {current}/{total_experiments}")
                
                config = config_fn(env_name)
                rewards, losses, elapsed = self.run_single_threaded_algorithm(
                    alg_name, config, game_name)
                
                self.results[game_name][alg_name] = {
                    'rewards': rewards,
                    'losses': losses,
                    'elapsed': elapsed
                }
                
                # Save intermediate results
                self.save_results()
        
        self.log(f"\n✓ All experiments completed!")
    
    def create_plots(self):
        """Create comprehensive plots"""
        self.log("\nGenerating plots...")
        
        colors = {
            'Basic DQN': '#1f77b4',
            'DQN + Dueling': '#ff7f0e', 
            'DQN + Priority': '#2ca02c'
        }
        
        # Main comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Alien and Ice Hockey RL Comparison ({self.episodes} episodes)', 
                     fontsize=16, fontweight='bold')
        
        for i, game_name in enumerate(self.game_names):
            # Episode rewards
            ax1 = axes[i, 0]
            for alg_name, color in colors.items():
                if alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    if rewards:
                        episodes = range(1, len(rewards) + 1)
                        ax1.plot(episodes, rewards, 'o-', color=color, 
                               linewidth=2, markersize=6, label=alg_name)
            
            ax1.set_title(f'{game_name} - Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bar chart of average performance
            ax2 = axes[i, 1]
            alg_names = []
            avg_rewards = []
            
            for alg_name in colors.keys():
                if alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    if rewards:
                        alg_names.append(alg_name)
                        avg_rewards.append(np.mean(rewards))
            
            if alg_names:
                bars = ax2.bar(range(len(alg_names)), avg_rewards, 
                             color=[colors[name] for name in alg_names])
                ax2.set_title(f'{game_name} - Average Performance')
                ax2.set_ylabel('Average Reward')
                ax2.set_xticks(range(len(alg_names)))
                ax2.set_xticklabels(alg_names, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, avg_rewards):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'comparison_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Main plot saved: {plot_path}")
        
        # Individual game plots
        for game_name in self.game_names:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for alg_name, color in colors.items():
                if alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    if rewards:
                        episodes = range(1, len(rewards) + 1)
                        ax.plot(episodes, rewards, 'o-', color=color,
                               linewidth=2, markersize=8, label=alg_name)
            
            ax.set_title(f'{game_name} - Learning Curves')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.output_dir, f'{game_name.lower()}_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log(f"✓ {game_name} plot saved: {plot_path}")
            plt.close()
        
        plt.show()
    
    def save_results(self):
        """Save results to JSON"""
        results_path = os.path.join(self.output_dir, 'results.json')
        
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
        
        self.log(f"✓ Results saved: {results_path}")
    
    def create_results_md(self):
        """Create results.md summary"""
        self.log("\nCreating results.md...")
        
        md_path = os.path.join(self.output_dir, 'results.md')
        
        with open(md_path, 'w') as f:
            # Header
            f.write(f"# RL Algorithms Comparison Results\n\n")
            f.write(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Episodes per Algorithm:** {self.episodes}\n")
            f.write(f"**Games Tested:** {', '.join(self.game_names)}\n")
            f.write(f"**Total Runtime:** {(time.time() - self.start_time):.1f} seconds\n\n")
            
            # Algorithms tested
            f.write("## Algorithms Tested\n\n")
            f.write("1. **Basic DQN** - Standard Deep Q-Network with Double Q-learning\n")
            f.write("2. **DQN + Dueling Networks** - DQN with separate value and advantage streams\n") 
            f.write("3. **DQN + Priority Replay** - DQN with prioritized experience replay\n\n")
            
            # Results summary
            f.write("## Results Summary\n\n")
            
            for game_name in self.game_names:
                f.write(f"### {game_name}\n\n")
                
                # Create results table
                f.write("| Algorithm | Avg Reward | Best Episode | Worst Episode | Training Time |\n")
                f.write("|-----------|------------|--------------|---------------|---------------|\n")
                
                # Sort by average performance
                game_results = []
                for alg_name in self.results[game_name]:
                    rewards = self.results[game_name][alg_name]['rewards']
                    elapsed = self.results[game_name][alg_name]['elapsed']
                    if rewards:
                        avg = np.mean(rewards)
                        best = max(rewards)
                        worst = min(rewards)
                        game_results.append((alg_name, avg, best, worst, elapsed))
                
                game_results.sort(key=lambda x: x[1], reverse=True)
                
                for i, (alg, avg, best, worst, elapsed) in enumerate(game_results):
                    rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    f.write(f"| {rank} {alg} | {avg:.2f} | {best:.1f} | {worst:.1f} | {elapsed:.1f}s |\n")
                
                f.write("\n")
                
                # Episode-by-episode results
                f.write(f"#### {game_name} - Episode Details\n\n")
                f.write("| Episode | Basic DQN | DQN + Dueling | DQN + Priority |\n")
                f.write("|---------|-----------|---------------|----------------|\n")
                
                max_episodes = max(len(self.results[game_name][alg]['rewards']) 
                                 for alg in self.results[game_name] if self.results[game_name][alg]['rewards'])
                
                for ep in range(max_episodes):
                    f.write(f"| {ep+1} |")
                    for alg in ['Basic DQN', 'DQN + Dueling', 'DQN + Priority']:
                        if alg in self.results[game_name]:
                            rewards = self.results[game_name][alg]['rewards']
                            if ep < len(rewards):
                                f.write(f" {rewards[ep]:.1f} |")
                            else:
                                f.write(" - |")
                        else:
                            f.write(" - |")
                    f.write("\n")
                
                f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find best algorithm overall
            overall_scores = {}
            for alg in ['Basic DQN', 'DQN + Dueling', 'DQN + Priority']:
                total_score = 0
                count = 0
                for game_name in self.game_names:
                    if alg in self.results[game_name]:
                        rewards = self.results[game_name][alg]['rewards']
                        if rewards:
                            total_score += np.mean(rewards)
                            count += 1
                if count > 0:
                    overall_scores[alg] = total_score / count
            
            if overall_scores:
                best_alg = max(overall_scores.keys(), key=lambda x: overall_scores[x])
                f.write(f"- **Best Overall Algorithm:** {best_alg} (avg score: {overall_scores[best_alg]:.2f})\n")
            
            # Game-specific observations
            for game_name in self.game_names:
                if game_name in self.results:
                    game_avgs = {}
                    for alg in self.results[game_name]:
                        rewards = self.results[game_name][alg]['rewards']
                        if rewards:
                            game_avgs[alg] = np.mean(rewards)
                    
                    if game_avgs:
                        best_game_alg = max(game_avgs.keys(), key=lambda x: game_avgs[x])
                        f.write(f"- **Best for {game_name}:** {best_game_alg} ({game_avgs[best_game_alg]:.2f} avg reward)\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `results.json` - Raw numerical results\n")
            f.write("- `experiment.log` - Detailed execution log\n")
            f.write("- `comparison_plot.png` - Main comparison visualization\n")
            f.write("- `alien_plot.png` - Alien game learning curves\n")
            f.write("- `icehockey_plot.png` - Ice Hockey game learning curves\n")
        
        self.log(f"✓ Results summary created: {md_path}")
    
    def print_summary(self):
        """Print final summary"""
        self.log(f"\n{'='*60}")
        self.log("EXPERIMENT COMPLETED SUCCESSFULLY!")
        self.log(f"{'='*60}")
        self.log(f"Total runtime: {(time.time() - self.start_time):.1f} seconds")
        self.log(f"Results directory: {self.output_dir}")
        
        for game_name in self.game_names:
            self.log(f"\n{game_name} Results:")
            for alg_name in self.results[game_name]:
                rewards = self.results[game_name][alg_name]['rewards']
                if rewards:
                    avg = np.mean(rewards)
                    self.log(f"  {alg_name:<20}: {avg:>8.2f} avg reward")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick Full RL Experiment')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per algorithm')
    args = parser.parse_args()
    
    try:
        experiment = QuickFullExperiment(episodes=args.episodes)
        experiment.run_all_experiments()
        experiment.create_plots()
        experiment.create_results_md()
        experiment.print_summary()
        
    except KeyboardInterrupt:
        experiment.log("\nExperiment interrupted by user")
    except Exception as e:
        experiment.log(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()