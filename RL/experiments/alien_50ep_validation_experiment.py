#!/usr/bin/env python3
"""
Alien-Only 50-Episode Validation Experiment
Runs 5 algorithms with proper time-matching and systematic recording
1. Basic DQN (50 episodes)
2. DQN + Dueling (50 episodes)  
3. DQN + Priority (50 episodes) - sets baseline time
4. Distributed + Priority (50 episodes)
5. Distributed + Priority (time-matched to DQN + Priority)
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


class AlienValidationExperiment:
    def __init__(self, episodes=50, output_dir='./experiments/results'):
        self.episodes = episodes
        self.env_name = 'ALE/Alien-v5'
        self.game_name = 'Alien'
        
        # Create unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"alien_validation_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.start_time = time.time()
        
        # Create log files
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        self.detailed_log = os.path.join(self.output_dir, 'detailed_results.json')
        
        self.log(f"ALIEN 50-EPISODE VALIDATION EXPERIMENT")
        self.log(f"Episodes per algorithm: {episodes}")
        self.log(f"Environment: {self.env_name}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*80)
    
    def log(self, message):
        """Log to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def create_basic_dqn_config(self) -> AgentConfig:
        """Create Basic DQN configuration"""
        config = AgentConfig()
        config.env_name = self.env_name
        config.use_dueling = False
        config.use_prioritized_replay = False
        
        # Standard settings for Alien
        config.memory_size = 10000
        config.batch_size = 32
        config.min_replay_size = 500
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.save_interval = 50000
        
        return config
    
    def create_dueling_dqn_config(self) -> AgentConfig:
        """Create Dueling DQN configuration"""
        config = self.create_basic_dqn_config()
        config.use_dueling = True
        return config
    
    def create_prioritized_dqn_config(self) -> AgentConfig:
        """Create Prioritized DQN configuration"""
        config = self.create_basic_dqn_config()
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        return config
    
    def create_distributed_config(self) -> DistributedAgentConfig:
        """Create Distributed + Priority configuration"""
        config = DistributedAgentConfig()
        config.env_name = self.env_name
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        
        # Optimized for Alien
        config.num_workers = 4
        config.memory_size = 20000
        config.batch_size = 64
        config.min_replay_size = 1000
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.save_interval = 50000
        
        return config
    
    def run_single_threaded(self, name: str, config) -> dict:
        """Run single-threaded algorithm and return detailed results"""
        self.log(f"\nRunning: {name}")
        self.log(f"  Config: buffer_size={config.memory_size}, batch_size={config.batch_size}")
        
        start_time = time.time()
        max_steps = 1500  # Limit steps per episode for Alien
        
        try:
            agent = DQNAgent(config)
            
            # Detailed tracking
            episode_rewards = []
            episode_steps = []
            episode_times = []
            losses = []
            epsilons = []
            
            for episode in range(self.episodes):
                ep_start = time.time()
                
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                agent.reset_hidden_state()
                
                episode_reward = 0
                episode_losses = []
                steps = 0
                
                done = False
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                    done = terminated or truncated
                    
                    next_state = agent.frame_stack.append(next_obs)
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    agent.steps_done += 1
                    steps += 1
                    
                    # Update Q-network
                    if agent.steps_done % agent.config.policy_update_interval == 0:
                        loss = agent.update_q_network()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    # Update target network
                    if agent.steps_done % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                
                # Update epsilon
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                # Update beta for prioritized replay
                if agent.config.use_prioritized_replay:
                    progress = min(episode / self.episodes, 1.0)
                    agent.priority_beta = agent.config.priority_beta_start + progress * (
                        agent.priority_beta_end - agent.config.priority_beta_start)
                    if hasattr(agent.replay_buffer, 'update_beta'):
                        agent.replay_buffer.update_beta(agent.priority_beta)
                
                # Record episode data
                episode_rewards.append(episode_reward)
                episode_steps.append(steps)
                episode_times.append(time.time() - ep_start)
                epsilons.append(agent.epsilon)
                
                if episode_losses:
                    losses.append(np.mean(episode_losses))
                else:
                    losses.append(0.0)
                
                # Progress logging every 10 episodes
                if episode % 10 == 0 or episode == self.episodes - 1:
                    elapsed = time.time() - start_time
                    avg_reward_recent = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                    self.log(f"    Episode {episode+1}/{self.episodes}: "
                            f"reward={episode_reward:.1f}, "
                            f"avg_recent={avg_reward_recent:.1f}, "
                            f"steps={steps}, "
                            f"epsilon={agent.epsilon:.3f}, "
                            f"time={elapsed:.1f}s")
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            results = {
                'algorithm': name,
                'episode_rewards': episode_rewards,
                'episode_steps': episode_steps,
                'episode_times': episode_times,
                'losses': losses,
                'epsilons': epsilons,
                'total_time': total_time,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'avg_steps': np.mean(episode_steps),
                'total_steps': sum(episode_steps),
                'episodes_per_minute': (self.episodes / total_time) * 60,
                'final_epsilon': epsilons[-1] if epsilons else 0.0
            }
            
            self.log(f"  ✓ {name} completed:")
            self.log(f"    Time: {total_time:.1f}s")
            self.log(f"    Avg reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            self.log(f"    Max/Min reward: {results['max_reward']:.0f}/{results['min_reward']:.0f}")
            self.log(f"    Throughput: {results['episodes_per_minute']:.2f} ep/min")
            
            return results
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            return None
    
    def run_distributed_fixed(self, name: str, config) -> dict:
        """Run distributed algorithm for fixed episodes"""
        self.log(f"\nRunning: {name}")
        self.log(f"  Config: workers={config.num_workers}, buffer_size={config.memory_size}, batch_size={config.batch_size}")
        
        start_time = time.time()
        
        try:
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            
            # Run training
            results = agent.train_distributed(total_episodes=self.episodes)
            
            total_time = time.time() - start_time
            
            # Extract detailed results
            env_stats = results.get('env_stats', {})
            training_stats = results.get('training_stats', {})
            
            # Extract episode rewards
            episode_rewards = []
            if 'worker_stats' in env_stats:
                for worker_stat in env_stats['worker_stats']:
                    worker_rewards = worker_stat.get('episode_rewards', [])
                    episode_rewards.extend(worker_rewards)
            
            # If no detailed rewards, use average
            if not episode_rewards and 'overall_avg_reward' in env_stats:
                avg = env_stats['overall_avg_reward']
                total_eps = env_stats.get('total_episodes', self.episodes)
                episode_rewards = [avg] * min(total_eps, self.episodes)
            
            # Ensure we have exactly 50 episodes
            if len(episode_rewards) < self.episodes:
                last_reward = episode_rewards[-1] if episode_rewards else 0
                episode_rewards.extend([last_reward] * (self.episodes - len(episode_rewards)))
            episode_rewards = episode_rewards[:self.episodes]
            
            # Extract losses
            losses = training_stats.get('losses', [])
            
            # Build results
            dist_results = {
                'algorithm': name,
                'episode_rewards': episode_rewards,
                'losses': losses,
                'total_time': total_time,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'total_episodes': len(episode_rewards),
                'episodes_per_minute': (len(episode_rewards) / total_time) * 60,
                'num_workers': config.num_workers,
                'total_steps': env_stats.get('total_steps', 0)
            }
            
            self.log(f"  ✓ {name} completed:")
            self.log(f"    Time: {total_time:.1f}s")
            self.log(f"    Avg reward: {dist_results['avg_reward']:.2f} ± {dist_results['std_reward']:.2f}")
            self.log(f"    Max/Min reward: {dist_results['max_reward']:.0f}/{dist_results['min_reward']:.0f}")
            self.log(f"    Throughput: {dist_results['episodes_per_minute']:.2f} ep/min")
            
            return dist_results
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            return None
    
    def run_distributed_timed(self, name: str, config, target_time: float) -> dict:
        """Run distributed algorithm for target time"""
        self.log(f"\nRunning: {name} (time-matched to {target_time:.1f}s)")
        self.log(f"  Config: workers={config.num_workers}, buffer_size={config.memory_size}, batch_size={config.batch_size}")
        
        start_time = time.time()
        
        try:
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            
            all_episode_rewards = []
            all_losses = []
            total_episodes_run = 0
            
            # Run in chunks until time limit
            while (time.time() - start_time) < target_time:
                remaining_time = target_time - (time.time() - start_time)
                if remaining_time < 5:  # Stop if less than 5 seconds left
                    break
                
                # Determine chunk size based on remaining time
                chunk_episodes = min(20, max(5, int(remaining_time / 20)))
                
                chunk_start = time.time()
                results = agent.train_distributed(total_episodes=chunk_episodes)
                chunk_time = time.time() - chunk_start
                
                # Extract results
                env_stats = results.get('env_stats', {})
                training_stats = results.get('training_stats', {})
                
                # Extract episode rewards
                chunk_rewards = []
                if 'worker_stats' in env_stats:
                    for worker_stat in env_stats['worker_stats']:
                        worker_rewards = worker_stat.get('episode_rewards', [])
                        chunk_rewards.extend(worker_rewards)
                
                # Fallback to average if no detailed rewards
                if not chunk_rewards and 'overall_avg_reward' in env_stats:
                    avg = env_stats['overall_avg_reward']
                    eps = env_stats.get('total_episodes', chunk_episodes)
                    chunk_rewards = [avg] * eps
                
                all_episode_rewards.extend(chunk_rewards)
                
                # Extract losses
                chunk_losses = training_stats.get('losses', [])
                all_losses.extend(chunk_losses)
                
                total_episodes_run += len(chunk_rewards)
                
                self.log(f"    Chunk: {len(chunk_rewards)} episodes in {chunk_time:.1f}s, "
                        f"avg reward: {np.mean(chunk_rewards):.2f}, "
                        f"total episodes: {total_episodes_run}")
            
            total_time = time.time() - start_time
            
            # Build results
            dist_results = {
                'algorithm': name,
                'episode_rewards': all_episode_rewards,
                'losses': all_losses,
                'total_time': total_time,
                'target_time': target_time,
                'avg_reward': np.mean(all_episode_rewards) if all_episode_rewards else 0,
                'std_reward': np.std(all_episode_rewards) if all_episode_rewards else 0,
                'max_reward': np.max(all_episode_rewards) if all_episode_rewards else 0,
                'min_reward': np.min(all_episode_rewards) if all_episode_rewards else 0,
                'total_episodes': len(all_episode_rewards),
                'episodes_per_minute': (len(all_episode_rewards) / total_time) * 60 if total_time > 0 else 0,
                'num_workers': config.num_workers
            }
            
            self.log(f"  ✓ {name} completed:")
            self.log(f"    Time: {total_time:.1f}s (target was {target_time:.1f}s)")
            self.log(f"    Episodes: {dist_results['total_episodes']}")
            self.log(f"    Avg reward: {dist_results['avg_reward']:.2f} ± {dist_results['std_reward']:.2f}")
            self.log(f"    Max/Min reward: {dist_results['max_reward']:.0f}/{dist_results['min_reward']:.0f}")
            self.log(f"    Throughput: {dist_results['episodes_per_minute']:.2f} ep/min")
            
            return dist_results
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            return None
    
    def run_all_experiments(self):
        """Run all 5 algorithms systematically"""
        self.log("\n" + "="*80)
        self.log("STARTING EXPERIMENTS")
        self.log("="*80)
        
        algorithms = [
            ('Basic DQN', 'single', self.create_basic_dqn_config),
            ('DQN + Dueling', 'single', self.create_dueling_dqn_config),
            ('DQN + Priority', 'single', self.create_prioritized_dqn_config),
            ('Distributed + Priority (50ep)', 'distributed_fixed', self.create_distributed_config),
            ('Distributed + Priority (time-matched)', 'distributed_timed', self.create_distributed_config),
        ]
        
        priority_time = None  # Will be set by DQN + Priority
        
        for i, (alg_name, alg_type, config_fn) in enumerate(algorithms):
            self.log(f"\n{'='*80}")
            self.log(f"Algorithm {i+1}/5: {alg_name}")
            self.log(f"{'='*80}")
            
            config = config_fn()
            
            if alg_type == 'single':
                result = self.run_single_threaded(alg_name, config)
                
                # Set baseline time from DQN + Priority
                if alg_name == 'DQN + Priority' and result:
                    priority_time = result['total_time']
                    self.log(f"\n  → Baseline time set: {priority_time:.1f}s")
            
            elif alg_type == 'distributed_fixed':
                result = self.run_distributed_fixed(alg_name, config)
            
            elif alg_type == 'distributed_timed':
                if priority_time is None:
                    self.log("  Warning: No baseline time available, using 600s default")
                    priority_time = 600
                result = self.run_distributed_timed(alg_name, config, priority_time)
            
            if result:
                self.results[alg_name] = result
                self.save_results()  # Save after each algorithm completes
            else:
                self.log(f"  Warning: {alg_name} failed to produce results")
    
    def save_results(self):
        """Save all results to JSON"""
        # Save detailed results
        with open(self.detailed_log, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for alg_name, data in self.results.items():
                json_results[alg_name] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in data.items()
                }
            json.dump(json_results, f, indent=2)
        
        self.log(f"  Saved detailed results to: {self.detailed_log}")
    
    def create_summary_report(self):
        """Create comprehensive summary report"""
        summary_path = os.path.join(self.output_dir, 'summary_report.md')
        
        with open(summary_path, 'w') as f:
            f.write("# Alien 50-Episode Validation Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Runtime:** {(time.time() - self.start_time)/60:.1f} minutes\n")
            f.write(f"**Environment:** {self.env_name}\n")
            f.write(f"**Episodes per Algorithm:** {self.episodes}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Algorithm | Episodes | Avg Reward | Std Dev | Max | Min | Time (s) | Throughput (ep/min) |\n")
            f.write("|-----------|----------|------------|---------|-----|-----|----------|--------------------|\n")
            
            for alg_name, data in self.results.items():
                f.write(f"| {alg_name} | {data.get('total_episodes', self.episodes)} | "
                       f"{data['avg_reward']:.2f} | {data['std_reward']:.2f} | "
                       f"{data['max_reward']:.0f} | {data['min_reward']:.0f} | "
                       f"{data['total_time']:.1f} | {data['episodes_per_minute']:.2f} |\n")
            
            # Performance comparison
            f.write("\n## Performance Analysis\n\n")
            
            # Find best reward
            best_reward_alg = max(self.results.items(), key=lambda x: x[1]['avg_reward'])
            f.write(f"**Best Average Reward:** {best_reward_alg[0]} ({best_reward_alg[1]['avg_reward']:.2f})\n\n")
            
            # Find fastest
            fastest_alg = max(self.results.items(), key=lambda x: x[1]['episodes_per_minute'])
            f.write(f"**Fastest Training:** {fastest_alg[0]} ({fastest_alg[1]['episodes_per_minute']:.2f} ep/min)\n\n")
            
            # Speedup analysis
            if 'Basic DQN' in self.results and 'Distributed + Priority (50ep)' in self.results:
                basic_throughput = self.results['Basic DQN']['episodes_per_minute']
                dist_throughput = self.results['Distributed + Priority (50ep)']['episodes_per_minute']
                speedup = dist_throughput / basic_throughput
                f.write(f"**Distributed Speedup:** {speedup:.1f}x over Basic DQN\n\n")
            
            # Time-matched comparison
            if 'DQN + Priority' in self.results and 'Distributed + Priority (time-matched)' in self.results:
                priority_reward = self.results['DQN + Priority']['avg_reward']
                dist_timed_reward = self.results['Distributed + Priority (time-matched)']['avg_reward']
                dist_timed_episodes = self.results['Distributed + Priority (time-matched)'].get('total_episodes', 0)
                
                f.write("## Time-Matched Comparison\n\n")
                f.write(f"**DQN + Priority:**\n")
                f.write(f"- Episodes: {self.episodes}\n")
                f.write(f"- Avg Reward: {priority_reward:.2f}\n")
                f.write(f"- Time: {self.results['DQN + Priority']['total_time']:.1f}s\n\n")
                
                f.write(f"**Distributed + Priority (time-matched):**\n")
                f.write(f"- Episodes: {dist_timed_episodes}\n")
                f.write(f"- Avg Reward: {dist_timed_reward:.2f}\n")
                f.write(f"- Time: {self.results['Distributed + Priority (time-matched)']['total_time']:.1f}s\n")
                f.write(f"- Episode Advantage: {dist_timed_episodes - self.episodes} more episodes in same time\n\n")
            
            f.write("## Conclusions\n\n")
            f.write("- Basic DQN performance compared to complex variants\n")
            f.write("- Distributed training speedup effectiveness\n")
            f.write("- Time-matched distributed training episode count advantage\n")
        
        self.log(f"\nSummary report saved to: {summary_path}")
    
    def create_visualization(self):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Average Rewards Comparison
            ax1 = axes[0, 0]
            alg_names = list(self.results.keys())
            avg_rewards = [self.results[alg]['avg_reward'] for alg in alg_names]
            std_rewards = [self.results[alg]['std_reward'] for alg in alg_names]
            
            x_pos = np.arange(len(alg_names))
            ax1.bar(x_pos, avg_rewards, yerr=std_rewards, capsize=5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(alg_names, rotation=45, ha='right')
            ax1.set_ylabel('Average Reward')
            ax1.set_title('Average Rewards by Algorithm')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Training Time Comparison
            ax2 = axes[0, 1]
            times = [self.results[alg]['total_time'] for alg in alg_names]
            ax2.bar(x_pos, times)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(alg_names, rotation=45, ha='right')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Training Time by Algorithm')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Throughput Comparison
            ax3 = axes[1, 0]
            throughputs = [self.results[alg]['episodes_per_minute'] for alg in alg_names]
            ax3.bar(x_pos, throughputs)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(alg_names, rotation=45, ha='right')
            ax3.set_ylabel('Episodes per Minute')
            ax3.set_title('Training Throughput by Algorithm')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Learning Curves (for single-threaded algorithms)
            ax4 = axes[1, 1]
            for alg_name in ['Basic DQN', 'DQN + Dueling', 'DQN + Priority']:
                if alg_name in self.results and 'episode_rewards' in self.results[alg_name]:
                    rewards = self.results[alg_name]['episode_rewards']
                    # Calculate moving average
                    window = 5
                    if len(rewards) >= window:
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        episodes = np.arange(window-1, len(rewards))
                        ax4.plot(episodes, moving_avg, label=alg_name, alpha=0.8)
            
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward (5-episode moving avg)')
            ax4.set_title('Learning Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'results_visualization.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            self.log(f"Visualization saved to: {plot_path}")
            
        except Exception as e:
            self.log(f"Could not create visualization: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Alien 50-Episode Validation Experiment')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per algorithm')
    args = parser.parse_args()
    
    # Run experiment
    experiment = AlienValidationExperiment(episodes=args.episodes)
    experiment.run_all_experiments()
    
    # Generate reports
    experiment.create_summary_report()
    experiment.create_visualization()
    
    # Final summary
    experiment.log("\n" + "="*80)
    experiment.log("EXPERIMENT COMPLETED!")
    experiment.log(f"Results directory: {experiment.output_dir}")
    experiment.log(f"Total time: {(time.time() - experiment.start_time)/60:.1f} minutes")
    experiment.log("="*80)


if __name__ == '__main__':
    main()