#!/usr/bin/env python3
"""
3-Algorithm 20-Episode Experiment with Paper's Learning Rate
Compares: Basic DQN, DQN + Priority, Distributed + Priority (time-matched)
Learning Rate: 2.5e-4 (from Prioritized Experience Replay paper)
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


class PaperLearningRateExperiment:
    def __init__(self, episodes=20, output_dir='./experiments/results'):
        self.episodes = episodes
        self.env_name = 'ALE/Alien-v5'
        self.paper_lr = 2.5e-4  # Learning rate from the paper
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"paper_lr_20ep_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.start_time = time.time()
        
        # Create log file
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log("="*80)
        self.log("3-ALGORITHM EXPERIMENT WITH PAPER'S LEARNING RATE")
        self.log("="*80)
        self.log(f"Episodes per algorithm: {episodes}")
        self.log(f"Learning rate: {self.paper_lr} (2.5e-4)")
        self.log(f"Environment: {self.env_name}")
        self.log(f"Algorithms: Basic DQN, DQN + Priority, Distributed + Priority")
        self.log(f"Output directory: {self.output_dir}")
        self.log("="*80)
    
    def log(self, message):
        """Log to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def create_basic_dqn_config(self) -> AgentConfig:
        """Create Basic DQN configuration with paper's learning rate"""
        config = AgentConfig()
        config.env_name = self.env_name
        config.use_dueling = False
        config.use_prioritized_replay = False
        
        # Paper's learning rate
        config.learning_rate = self.paper_lr  # 2.5e-4
        
        # Standard settings
        config.memory_size = 10000
        config.batch_size = 32
        config.min_replay_size = 500
        config.target_update_freq = 500
        config.save_interval = 50000
        
        return config
    
    def create_prioritized_dqn_config(self) -> AgentConfig:
        """Create Prioritized DQN configuration with paper's learning rate"""
        config = self.create_basic_dqn_config()
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6  # From paper
        config.priority_beta_start = 0.4  # From paper
        config.priority_beta_end = 1.0
        return config
    
    def create_distributed_config(self) -> DistributedAgentConfig:
        """Create Distributed + Priority configuration with paper's learning rate"""
        config = DistributedAgentConfig()
        config.env_name = self.env_name
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        
        # Paper's learning rate
        config.learning_rate = self.paper_lr  # 2.5e-4
        
        # Distributed settings
        config.num_workers = 4
        config.memory_size = 20000
        config.batch_size = 64
        config.min_replay_size = 1000
        config.target_update_freq = 500
        config.save_interval = 50000
        
        return config
    
    def run_single_threaded(self, name: str, config) -> dict:
        """Run single-threaded algorithm"""
        self.log(f"\nRunning: {name}")
        self.log(f"  Config: lr={config.learning_rate}, buffer={config.memory_size}, batch={config.batch_size}")
        
        start_time = time.time()
        max_steps = 1500  # Limit steps per episode
        
        try:
            agent = DQNAgent(config)
            
            # Tracking
            episode_rewards = []
            episode_steps = []
            losses = []
            
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
                
                if episode_losses:
                    losses.append(np.mean(episode_losses))
                
                # Progress logging every 5 episodes
                if episode % 5 == 0 or episode == self.episodes - 1:
                    elapsed = time.time() - start_time
                    avg_reward_recent = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
                    self.log(f"    Episode {episode+1}/{self.episodes}: "
                            f"reward={episode_reward:.1f}, "
                            f"avg_recent={avg_reward_recent:.1f}, "
                            f"steps={steps}, "
                            f"epsilon={agent.epsilon:.3f}, "
                            f"time={elapsed:.1f}s")
            
            total_time = time.time() - start_time
            total_steps = sum(episode_steps)
            
            # Calculate statistics
            results = {
                'algorithm': name,
                'learning_rate': config.learning_rate,
                'episode_rewards': episode_rewards,
                'episode_steps': episode_steps,
                'losses': losses,
                'total_time': total_time,
                'total_steps': total_steps,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'avg_steps_per_episode': np.mean(episode_steps),
                'episodes_per_minute': (self.episodes / total_time) * 60
            }
            
            self.log(f"  ✓ {name} completed:")
            self.log(f"    Time: {total_time:.1f}s")
            self.log(f"    Total steps: {total_steps}")
            self.log(f"    Avg reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            self.log(f"    Throughput: {results['episodes_per_minute']:.2f} ep/min")
            
            return results
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            return None
    
    def run_distributed_timematched(self, name: str, config, target_time: float) -> dict:
        """Run distributed training for target time"""
        self.log(f"\nRunning: {name} (time-matched to {target_time:.1f}s)")
        self.log(f"  Config: lr={config.learning_rate}, workers={config.num_workers}, buffer={config.memory_size}")
        
        # Estimate episodes based on observed throughput
        estimated_episodes = int(68.7 * (target_time / 60))  # ~68.7 ep/min from previous runs
        self.log(f"  Estimated episodes for {target_time:.1f}s: {estimated_episodes}")
        
        start_time = time.time()
        
        try:
            # Create agent
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            
            # Run training for estimated episodes with time limit
            results = agent.train_distributed(
                total_episodes=estimated_episodes,
                max_time_seconds=target_time
            )
            
            total_time = time.time() - start_time
            
            # Extract results
            env_stats = results.get('env_stats', {})
            training_stats = results.get('training_stats', {})
            
            actual_episodes = env_stats.get('total_episodes', 0)
            avg_reward = env_stats.get('overall_avg_reward', 0)
            total_steps = env_stats.get('total_steps', 0)
            
            # Get per-worker stats
            worker_stats = env_stats.get('worker_stats', [])
            worker_episodes = [w.get('total_episodes', 0) for w in worker_stats]
            worker_rewards = [w.get('avg_reward', 0) for w in worker_stats]
            
            # Build results
            dist_results = {
                'algorithm': name,
                'learning_rate': config.learning_rate,
                'target_time': target_time,
                'actual_time': total_time,
                'estimated_episodes': estimated_episodes,
                'actual_episodes': actual_episodes,
                'total_steps': total_steps,
                'avg_reward': avg_reward,
                'episodes_per_minute': (actual_episodes / total_time) * 60 if total_time > 0 else 0,
                'num_workers': config.num_workers,
                'worker_episodes': worker_episodes,
                'worker_avg_rewards': worker_rewards
            }
            
            self.log(f"  ✓ {name} completed:")
            self.log(f"    Time: {total_time:.1f}s (target was {target_time:.1f}s)")
            self.log(f"    Episodes: {actual_episodes}")
            self.log(f"    Total steps: {total_steps}")
            self.log(f"    Avg reward: {avg_reward:.2f}")
            self.log(f"    Throughput: {dist_results['episodes_per_minute']:.2f} ep/min")
            
            return dist_results
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            return None
    
    def run_experiment(self):
        """Run the 3-algorithm experiment"""
        self.log("\n" + "="*80)
        self.log("STARTING EXPERIMENTS")
        self.log("="*80)
        
        priority_time = None
        
        # 1. Basic DQN
        self.log("\n" + "-"*80)
        self.log("Algorithm 1/3: Basic DQN")
        self.log("-"*80)
        config = self.create_basic_dqn_config()
        result = self.run_single_threaded("Basic DQN", config)
        if result:
            self.results["Basic DQN"] = result
            self.save_results()
        
        # 2. DQN + Priority
        self.log("\n" + "-"*80)
        self.log("Algorithm 2/3: DQN + Priority")
        self.log("-"*80)
        config = self.create_prioritized_dqn_config()
        result = self.run_single_threaded("DQN + Priority", config)
        if result:
            self.results["DQN + Priority"] = result
            priority_time = result['total_time']
            self.log(f"\n  → Baseline time set: {priority_time:.1f}s")
            self.save_results()
        
        # 3. Distributed + Priority (time-matched)
        if priority_time:
            self.log("\n" + "-"*80)
            self.log("Algorithm 3/3: Distributed + Priority (time-matched)")
            self.log("-"*80)
            config = self.create_distributed_config()
            result = self.run_distributed_timematched(
                "Distributed + Priority (time-matched)", 
                config, 
                priority_time
            )
            if result:
                self.results["Distributed + Priority (time-matched)"] = result
                self.save_results()
    
    def save_results(self):
        """Save results to JSON"""
        results_file = os.path.join(self.output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log(f"  Results saved to: {results_file}")
    
    def create_final_report(self):
        """Create final comparison report"""
        report_file = os.path.join(self.output_dir, 'final_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# 3-Algorithm 20-Episode Experiment with Paper's Learning Rate\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Runtime:** {(time.time() - self.start_time)/60:.1f} minutes\n")
            f.write(f"**Learning Rate:** {self.paper_lr} (2.5e-4 from paper)\n")
            f.write(f"**Episodes per Algorithm:** {self.episodes}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Algorithm | Episodes | Total Steps | Avg Reward | Time (s) | Throughput (ep/min) |\n")
            f.write("|-----------|----------|-------------|------------|----------|--------------------|\n")
            
            for alg_name, data in self.results.items():
                if 'actual_episodes' in data:  # Distributed
                    episodes = data['actual_episodes']
                    steps = data['total_steps']
                else:  # Single-threaded
                    episodes = self.episodes
                    steps = data['total_steps']
                
                f.write(f"| {alg_name} | {episodes} | {steps} | "
                       f"{data.get('avg_reward', 0):.2f} | "
                       f"{data.get('total_time', data.get('actual_time', 0)):.1f} | "
                       f"{data.get('episodes_per_minute', 0):.2f} |\n")
            
            # Comparison section
            if "DQN + Priority" in self.results and "Distributed + Priority (time-matched)" in self.results:
                dqn_p = self.results["DQN + Priority"]
                dist_p = self.results["Distributed + Priority (time-matched)"]
                
                f.write("\n## Time-Matched Comparison\n\n")
                f.write(f"**DQN + Priority:**\n")
                f.write(f"- Episodes: {self.episodes}\n")
                f.write(f"- Steps: {dqn_p['total_steps']}\n")
                f.write(f"- Avg Reward: {dqn_p['avg_reward']:.2f}\n")
                f.write(f"- Time: {dqn_p['total_time']:.1f}s\n\n")
                
                f.write(f"**Distributed + Priority (time-matched):**\n")
                f.write(f"- Episodes: {dist_p['actual_episodes']}\n")
                f.write(f"- Steps: {dist_p['total_steps']}\n")
                f.write(f"- Avg Reward: {dist_p['avg_reward']:.2f}\n")
                f.write(f"- Time: {dist_p['actual_time']:.1f}s\n")
                f.write(f"- Episode Advantage: {dist_p['actual_episodes'] - self.episodes} more episodes\n")
                f.write(f"- Step Advantage: {dist_p['total_steps'] - dqn_p['total_steps']} more steps\n")
                f.write(f"- Speedup: {dist_p['episodes_per_minute'] / dqn_p['episodes_per_minute']:.1f}x\n\n")
            
            # Learning rate comparison
            f.write("## Learning Rate Impact\n\n")
            f.write("Comparison with previous experiments (1e-4 learning rate):\n")
            f.write("- Paper's LR (2.5e-4) is 2.5x larger\n")
            f.write("- Should lead to faster initial learning\n")
            f.write("- May show more variance in rewards\n")
        
        self.log(f"\nFinal report saved to: {report_file}")


def main():
    # Create and run experiment
    experiment = PaperLearningRateExperiment(episodes=20)
    
    # Run the 3 algorithms
    experiment.run_experiment()
    
    # Create final report
    experiment.create_final_report()
    
    # Print summary
    experiment.log("\n" + "="*80)
    experiment.log("EXPERIMENT COMPLETED!")
    experiment.log("="*80)
    
    if experiment.results:
        experiment.log("\nFinal Summary:")
        for alg_name, data in experiment.results.items():
            if 'actual_episodes' in data:  # Distributed
                experiment.log(f"- {alg_name}: {data['actual_episodes']} episodes, "
                             f"{data['avg_reward']:.2f} reward, {data['actual_time']:.1f}s")
            else:  # Single-threaded
                experiment.log(f"- {alg_name}: 20 episodes, "
                             f"{data['avg_reward']:.2f} reward, {data['total_time']:.1f}s")


if __name__ == '__main__':
    main()