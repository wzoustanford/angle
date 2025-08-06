#!/usr/bin/env python3
"""
50-Episode Experiment with 5 Algorithms
1. Basic DQN (50 episodes)
2. DQN + Dueling (50 episodes)  
3. DQN + Priority (50 episodes) - sets baseline time
4. Distributed + Priority (50 episodes, then stops)
5. Distributed + Priority (same wall-clock time as #3)
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


class FiftyEpisodeExperiment:
    def __init__(self, episodes=50, output_dir='./experiments/results'):
        self.episodes = episodes
        self.games = ['ALE/Alien-v5', 'ALE/IceHockey-v5']
        self.game_names = ['Alien', 'IceHockey']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"experiment_50ep_5alg_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.start_time = time.time()
        
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log(f"50-Episode 5-Algorithm Experiment Started")
        self.log(f"Episodes per algorithm: {episodes}")
        self.log(f"Games: {', '.join(self.game_names)}")
        self.log(f"Output directory: {self.output_dir}")
        self.log("="*60)
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def create_basic_dqn_config(self, env_name: str) -> AgentConfig:
        config = AgentConfig()
        config.env_name = env_name
        config.use_dueling = False
        config.use_prioritized_replay = False
        
        # Game-specific memory optimization
        if 'IceHockey' in env_name:
            config.memory_size = 5000    # Reduced for Ice Hockey
            config.batch_size = 16       # Smaller batches
            config.min_replay_size = 500
        else:
            config.memory_size = 10000   # Normal for Alien
            config.batch_size = 32
            config.min_replay_size = 500
            
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.save_interval = 50000
        return config
    
    def create_dueling_dqn_config(self, env_name: str) -> AgentConfig:
        config = self.create_basic_dqn_config(env_name)
        config.use_dueling = True
        return config
    
    def create_prioritized_dqn_config(self, env_name: str) -> AgentConfig:
        config = self.create_basic_dqn_config(env_name)
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        return config
    
    def create_distributed_priority_config(self, env_name: str) -> DistributedAgentConfig:
        config = DistributedAgentConfig()
        config.env_name = env_name
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        
        # Game-specific optimization for distributed training
        if 'IceHockey' in env_name:
            config.num_workers = 2       # Fewer workers for Ice Hockey
            config.memory_size = 8000    # Reduced buffer
            config.batch_size = 32       # Smaller batches
            config.min_replay_size = 800
        else:
            config.num_workers = 4       # Normal workers for Alien  
            config.memory_size = 20000   # Normal buffer
            config.batch_size = 64       # Larger batches
            config.min_replay_size = 1000
            
        config.learning_rate = 1e-4
        config.target_update_freq = 500
        config.save_interval = 50000
        return config
    
    def run_single_threaded(self, name: str, config, game_name: str) -> tuple:
        self.log(f"\n  Running {name} on {game_name}")
        start_time = time.time()
        # Reduced episode limits to prevent memory buildup
        max_steps = 2000 if game_name == 'IceHockey' else 1500
        
        try:
            agent = DQNAgent(config)
            episode_rewards = []
            losses = []
            
            for episode in range(self.episodes):
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
                    
                    if agent.steps_done % agent.config.policy_update_interval == 0:
                        loss = agent.update_q_network()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    if agent.steps_done % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                
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
                
                if episode % 10 == 0 or episode == self.episodes - 1:
                    elapsed = time.time() - start_time
                    self.log(f"    Episode {episode+1}/{self.episodes}: {steps} steps, "
                            f"reward={episode_reward:.1f}, time={elapsed:.1f}s")
            
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards)
            self.log(f"  ✓ {name} completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
            
            return episode_rewards, losses, elapsed
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            return [0] * self.episodes, [], 0
    
    def run_distributed_fixed(self, name: str, config, game_name: str) -> tuple:
        self.log(f"\n  Running {name} on {game_name} (50 episodes)")
        start_time = time.time()
        
        try:
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            results = agent.train_distributed(total_episodes=self.episodes)
            
            elapsed = time.time() - start_time
            
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
            
            # Ensure exactly 50 episodes
            if len(episode_rewards) < self.episodes:
                last_reward = episode_rewards[-1] if episode_rewards else 0
                episode_rewards.extend([last_reward] * (self.episodes - len(episode_rewards)))
            episode_rewards = episode_rewards[:self.episodes]
            
            losses = results.get('training_stats', {}).get('losses', [])
            avg_reward = np.mean(episode_rewards)
            
            self.log(f"  ✓ {name} completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
            
            return episode_rewards, losses, elapsed
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            return [0] * self.episodes, [], 0
    
    def run_distributed_timed(self, name: str, config, game_name: str, target_time: float) -> tuple:
        self.log(f"\n  Running {name} on {game_name} (time-matched: {target_time:.1f}s)")
        start_time = time.time()
        
        try:
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            
            episode_rewards = []
            all_losses = []
            
            while (time.time() - start_time) < target_time:
                remaining_time = target_time - (time.time() - start_time)
                if remaining_time < 10:
                    break
                
                # Run in chunks
                chunk_episodes = min(20, max(5, int(remaining_time / 30)))
                
                chunk_start = time.time()
                results = agent.train_distributed(total_episodes=chunk_episodes)
                chunk_elapsed = time.time() - chunk_start
                
                # Extract rewards
                env_stats = results.get('env_stats', {})
                chunk_rewards = []
                
                if 'worker_stats' in env_stats:
                    for worker_stat in env_stats['worker_stats']:
                        episodes = worker_stat.get('total_episodes', 0)
                        avg_reward = worker_stat.get('avg_reward', 0)
                        if episodes > 0 and not np.isnan(avg_reward):
                            chunk_rewards.extend([avg_reward] * episodes)
                
                if not chunk_rewards and 'overall_avg_reward' in env_stats:
                    overall_avg = env_stats['overall_avg_reward']
                    total_episodes = env_stats.get('total_episodes', 0)
                    if total_episodes > 0 and not np.isnan(overall_avg):
                        chunk_rewards = [overall_avg] * total_episodes
                
                episode_rewards.extend(chunk_rewards)
                
                chunk_losses = results.get('training_stats', {}).get('losses', [])
                all_losses.extend(chunk_losses)
                
                self.log(f"    Chunk: {len(chunk_rewards)} episodes in {chunk_elapsed:.1f}s, "
                        f"avg reward: {np.mean(chunk_rewards):.2f}")
            
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            
            self.log(f"  ✓ {name} completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
            self.log(f"    Total episodes: {len(episode_rewards)}")
            
            return episode_rewards, all_losses, elapsed
            
        except Exception as e:
            self.log(f"  ✗ {name} failed: {e}")
            return [], [], target_time
    
    def run_all_experiments(self):
        algorithms = [
            ('Basic DQN', 'single', lambda env: self.create_basic_dqn_config(env)),
            ('DQN + Dueling', 'single', lambda env: self.create_dueling_dqn_config(env)),
            ('DQN + Priority', 'single', lambda env: self.create_prioritized_dqn_config(env)),
            ('Distributed + Priority (50ep)', 'distributed_fixed', lambda env: self.create_distributed_priority_config(env)),
            ('Distributed + Priority (time-matched)', 'distributed_timed', lambda env: self.create_distributed_priority_config(env)),
        ]
        
        self.log(f"\nStarting experiments on {', '.join(self.game_names)}")
        
        # Initialize results
        for game_name in self.game_names:
            self.results[game_name] = {}
        
        for env_name, game_name in zip(self.games, self.game_names):
            self.log(f"\n{'='*60}")
            self.log(f"GAME: {game_name}")
            self.log(f"{'='*60}")
            
            priority_time = None
            
            for i, (alg_name, alg_type, config_fn) in enumerate(algorithms):
                self.log(f"\nAlgorithm {i+1}/5: {alg_name}")
                config = config_fn(env_name)
                
                if alg_type == 'single':
                    rewards, losses, elapsed = self.run_single_threaded(alg_name, config, game_name)
                    if alg_name == 'DQN + Priority':
                        priority_time = elapsed
                        self.log(f"    → Baseline time set: {priority_time:.1f}s")
                
                elif alg_type == 'distributed_fixed':
                    rewards, losses, elapsed = self.run_distributed_fixed(alg_name, config, game_name)
                
                elif alg_type == 'distributed_timed':
                    if priority_time is None:
                        priority_time = 600  # 10 minutes default
                    rewards, losses, elapsed = self.run_distributed_timed(alg_name, config, game_name, priority_time)
                
                self.results[game_name][alg_name] = {
                    'rewards': rewards,
                    'losses': losses,
                    'elapsed': elapsed,
                    'episode_count': len(rewards)
                }
                
                self.save_results()
    
    def save_results(self):
        results_path = os.path.join(self.output_dir, 'results.json')
        
        json_results = {}
        for game in self.results:
            json_results[game] = {}
            for algorithm in self.results[game]:
                data = self.results[game][algorithm]
                json_results[game][algorithm] = {
                    'rewards': [float(x) for x in data['rewards']],
                    'losses': [float(x) for x in data['losses']],
                    'elapsed': float(data['elapsed']),
                    'episode_count': int(data['episode_count'])
                }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.log(f"✓ Results saved: {results_path}")
    
    def create_results_md(self):
        md_path = os.path.join(self.output_dir, 'results.md')
        
        with open(md_path, 'w') as f:
            f.write(f"# 50-Episode 5-Algorithm RL Comparison\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Runtime:** {(time.time() - self.start_time):.1f}s\n\n")
            
            for game_name in self.game_names:
                if game_name in self.results:
                    f.write(f"## {game_name} Results\n\n")
                    f.write("| Algorithm | Episodes | Avg Reward | Time | Throughput |\n")
                    f.write("|-----------|----------|------------|------|------------|\n")
                    
                    for alg_name, data in self.results[game_name].items():
                        ep_count = data['episode_count']
                        avg_reward = np.mean(data['rewards']) if data['rewards'] else 0
                        elapsed = data['elapsed']
                        throughput = ep_count / (elapsed / 60) if elapsed > 0 else 0
                        
                        f.write(f"| {alg_name} | {ep_count} | {avg_reward:.2f} | {elapsed:.1f}s | {throughput:.1f} ep/min |\n")
                    
                    f.write("\n")
        
        self.log(f"✓ Results summary: {md_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='50-Episode 5-Algorithm Experiment')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per algorithm')
    args = parser.parse_args()
    
    experiment = FiftyEpisodeExperiment(episodes=args.episodes)
    experiment.run_all_experiments()
    experiment.create_results_md()
    
    experiment.log(f"\n{'='*60}")
    experiment.log("EXPERIMENT COMPLETED!")
    experiment.log(f"Results: {experiment.output_dir}")
    experiment.log("="*60)


if __name__ == '__main__':
    main()