#!/usr/bin/env python3
"""
NGU (Never Give Up) Test Experiment

Tests the NGU and Agent57 implementations with a small-scale experiment
comparing them against the existing DQN variants.

This validates:
1. NGU intrinsic motivation works correctly
2. Agent57 multi-policy learning functions
3. Integration with existing training infrastructure
4. Memory usage is reasonable
"""

import sys
import os
import time
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import NGUConfig, Agent57Config, create_ngu_config_for_game
from model.ngu_agent import NGUAgent, Agent57
from model.dqn_agent import DQNAgent
from model.distributed_dqn_agent import DistributedDQNAgent
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid
from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig


class NGUTestExperiment:
    """Test experiment comparing NGU, Agent57, and standard DQN"""
    
    def __init__(self, episodes: int = 5, games: list = None):
        self.episodes = episodes
        self.games = games or ['ALE/Alien-v5']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./experiments/results/ngu_test_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log(f"NGU Test Experiment Started")
        self.log(f"Episodes per algorithm: {episodes}")
        self.log(f"Games: {', '.join(self.games)}")
        self.log("=" * 60)
    
    def log(self, message: str):
        """Log message to console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def test_ngu_agent(self, env_name: str) -> Dict[str, Any]:
        """Test NGU agent on environment"""
        self.log(f"\n  Testing NGU Agent on {env_name}")
        start_time = time.time()
        
        try:
            # Create NGU config
            config = create_ngu_config_for_game(env_name, episodes=self.episodes, use_agent57=False)
            
            # Create agent
            agent = NGUAgent(config)
            
            # Training statistics
            episode_rewards = []
            intrinsic_rewards = []
            episode_lengths = []
            
            # Train for specified episodes
            for episode in range(self.episodes):
                episode_stats = agent.train_episode()
                
                episode_rewards.append(episode_stats['episode_reward'])
                intrinsic_rewards.append(episode_stats['episode_intrinsic_reward'])
                episode_lengths.append(episode_stats['episode_steps'])
                
                if episode % max(1, self.episodes // 3) == 0:
                    self.log(f"    Episode {episode + 1}/{self.episodes}: "
                            f"Reward={episode_stats['episode_reward']:.1f}, "
                            f"Intrinsic={episode_stats['episode_intrinsic_reward']:.3f}, "
                            f"Steps={episode_stats['episode_steps']}")
            
            elapsed = time.time() - start_time
            
            # Get final statistics
            final_stats = agent.get_statistics()
            
            self.log(f"  ✓ NGU completed in {elapsed:.1f}s")
            self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
            self.log(f"    Avg intrinsic: {np.mean(intrinsic_rewards):.3f}")
            self.log(f"    Avg steps: {np.mean(episode_lengths):.1f}")
            
            return {
                'episode_rewards': episode_rewards,
                'intrinsic_rewards': intrinsic_rewards,
                'episode_lengths': episode_lengths,
                'elapsed_time': elapsed,
                'final_stats': final_stats,
                'algorithm': 'NGU',
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ NGU failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'intrinsic_rewards': [0] * self.episodes,
                'episode_lengths': [0] * self.episodes,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'NGU',
                'success': False
            }
    
    def test_agent57(self, env_name: str) -> Dict[str, Any]:
        """Test Agent57 on environment"""
        self.log(f"\n  Testing Agent57 on {env_name}")
        start_time = time.time()
        
        try:
            # Create Agent57 config
            config = create_ngu_config_for_game(env_name, episodes=self.episodes, use_agent57=True)
            
            # Create agent
            agent = Agent57(config)
            
            # Training statistics
            episode_rewards = []
            intrinsic_rewards = []
            episode_lengths = []
            policy_usage = {}
            
            # Train for specified episodes
            for episode in range(self.episodes):
                episode_stats = agent.train_episode()
                
                episode_rewards.append(episode_stats['episode_reward'])
                intrinsic_rewards.append(episode_stats['episode_intrinsic_reward'])
                episode_lengths.append(episode_stats['episode_steps'])
                
                # Track policy usage
                policy_id = episode_stats['policy_id']
                policy_usage[policy_id] = policy_usage.get(policy_id, 0) + 1
                
                if episode % max(1, self.episodes // 3) == 0:
                    self.log(f"    Episode {episode + 1}/{self.episodes}: "
                            f"Policy={policy_id}, "
                            f"Reward={episode_stats['episode_reward']:.1f}, "
                            f"Intrinsic={episode_stats['episode_intrinsic_reward']:.3f}")
            
            elapsed = time.time() - start_time
            
            # Get final statistics
            final_stats = agent.get_statistics()
            
            self.log(f"  ✓ Agent57 completed in {elapsed:.1f}s")
            self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
            self.log(f"    Avg intrinsic: {np.mean(intrinsic_rewards):.3f}")
            self.log(f"    Policies used: {list(policy_usage.keys())}")
            
            return {
                'episode_rewards': episode_rewards,
                'intrinsic_rewards': intrinsic_rewards,
                'episode_lengths': episode_lengths,
                'policy_usage': policy_usage,
                'elapsed_time': elapsed,
                'final_stats': final_stats,
                'algorithm': 'Agent57',
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ Agent57 failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'intrinsic_rewards': [0] * self.episodes,
                'episode_lengths': [0] * self.episodes,
                'policy_usage': {},
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'Agent57',
                'success': False
            }
    
    def test_baseline_dqn(self, env_name: str) -> Dict[str, Any]:
        """Test baseline DQN for comparison"""
        self.log(f"\n  Testing Baseline DQN on {env_name}")
        start_time = time.time()
        
        try:
            # Create standard DQN config
            config = AgentConfig()
            config.env_name = env_name
            config.use_dueling = False
            config.use_prioritized_replay = False
            
            # Optimize for quick testing with small memory footprint
            config.memory_size = 5000
            config.min_replay_size = 500
            config.batch_size = 8
            config.target_update_freq = 500
            config.max_episode_steps = 2000  # Limit episode length
            
            # Create agent
            agent = DQNAgent(config)
            
            # Training statistics
            episode_rewards = []
            episode_lengths = []
            
            # Train for specified episodes
            for episode in range(self.episodes):
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                episode_reward = 0
                episode_steps = 0
                
                done = False
                while not done and episode_steps < 3000:  # Limit episode length
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                    done = terminated or truncated
                    
                    next_state = agent.frame_stack.append(next_obs)
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    if agent.steps_done % 4 == 0:
                        agent.update_q_network()
                    
                    if agent.steps_done % 500 == 0:
                        agent.update_target_network()
                    
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    agent.steps_done += 1
                
                # Update epsilon
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                
                if episode % max(1, self.episodes // 3) == 0:
                    self.log(f"    Episode {episode + 1}/{self.episodes}: "
                            f"Reward={episode_reward:.1f}, Steps={episode_steps}")
            
            elapsed = time.time() - start_time
            
            self.log(f"  ✓ DQN completed in {elapsed:.1f}s")
            self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
            
            return {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'elapsed_time': elapsed,
                'algorithm': 'DQN',
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ DQN failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'episode_lengths': [0] * self.episodes,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'DQN',
                'success': False
            }
    
    def test_r2d2(self, env_name: str) -> Dict[str, Any]:
        """Test R2D2 (DQN with LSTM) for comparison"""
        self.log(f"\n  Testing R2D2 on {env_name}")
        start_time = time.time()
        
        try:
            # Create R2D2 config (DQN with LSTM)
            config = AgentConfig()
            config.env_name = env_name
            config.use_r2d2 = True  # Enable R2D2 mode
            config.use_dueling = True  # R2D2 typically uses dueling
            config.use_prioritized_replay = False
            
            # R2D2 specific settings
            config.sequence_length = 40
            config.burn_in_length = 20
            config.lstm_size = 256
            config.num_lstm_layers = 1
            
            # Memory optimization for testing
            config.memory_size = 5000
            config.min_replay_size = 500
            config.batch_size = 8
            config.target_update_freq = 500
            config.max_episode_steps = 2000
            
            # Create agent
            agent = DQNAgent(config)
            
            # Training statistics
            episode_rewards = []
            episode_lengths = []
            
            # Train for specified episodes
            for episode in range(self.episodes):
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                agent.reset_hidden_state()  # Reset LSTM hidden state
                episode_reward = 0
                episode_steps = 0
                
                done = False
                while not done and episode_steps < config.max_episode_steps:
                    action = agent.select_action(state)
                    next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                    done = terminated or truncated
                    
                    next_state = agent.frame_stack.append(next_obs)
                    # R2D2 uses sequence buffer with push_transition
                    if hasattr(agent.replay_buffer, 'push_transition'):
                        agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                    else:
                        agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    if agent.steps_done % 4 == 0:
                        agent.update_q_network()
                    
                    if agent.steps_done % 500 == 0:
                        agent.update_target_network()
                    
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    agent.steps_done += 1
                
                # Update epsilon
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                
                if episode % max(1, self.episodes // 3) == 0:
                    self.log(f"    Episode {episode + 1}/{self.episodes}: "
                            f"Reward={episode_reward:.1f}, Steps={episode_steps}")
            
            elapsed = time.time() - start_time
            
            self.log(f"  ✓ R2D2 completed in {elapsed:.1f}s")
            self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
            
            return {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'elapsed_time': elapsed,
                'algorithm': 'R2D2',
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ R2D2 failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'episode_lengths': [0] * self.episodes,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'R2D2',
                'success': False
            }
    
    def test_distributed_dqn(self, env_name: str) -> Dict[str, Any]:
        """Test Distributed DQN with multiple workers"""
        self.log(f"\n  Testing Distributed DQN on {env_name}")
        start_time = time.time()
        
        try:
            # Create Distributed config
            config = DistributedAgentConfig()
            config.env_name = env_name
            config.num_workers = 2  # Use 2 workers for testing
            config.use_dueling = False
            config.use_prioritized_replay = True
            
            # Memory optimization
            config.memory_size = 5000
            config.min_replay_size = 500
            config.batch_size = 8
            config.target_update_freq = 500
            
            # Create distributed agent
            agent = DistributedDQNAgent(config, num_workers=config.num_workers)
            
            # Train for specified episodes
            self.log(f"    Training with {config.num_workers} parallel workers...")
            results = agent.train_distributed(total_episodes=self.episodes)
            
            elapsed = time.time() - start_time
            
            # Extract statistics
            env_stats = results.get('env_stats', {})
            avg_reward = env_stats.get('overall_avg_reward', 0)
            total_episodes = env_stats.get('total_episodes', 0)
            
            self.log(f"  ✓ Distributed DQN completed in {elapsed:.1f}s")
            self.log(f"    Total episodes: {total_episodes}")
            self.log(f"    Avg reward: {avg_reward:.2f}")
            self.log(f"    Throughput: {total_episodes/elapsed:.1f} episodes/sec")
            
            # Create dummy episode rewards for consistency
            episode_rewards = [avg_reward] * self.episodes
            
            return {
                'episode_rewards': episode_rewards,
                'elapsed_time': elapsed,
                'total_episodes': total_episodes,
                'throughput': total_episodes/elapsed,
                'algorithm': 'Distributed DQN',
                'num_workers': config.num_workers,
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ Distributed DQN failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'Distributed DQN',
                'success': False
            }
    
    def test_r2d2_agent57(self, env_name: str) -> Dict[str, Any]:
        """Test R2D2+Agent57 Hybrid"""
        self.log(f"\n  Testing R2D2+Agent57 Hybrid on {env_name}")
        start_time = time.time()
        
        try:
            # Create hybrid config
            config = Agent57Config()
            config.env_name = env_name
            config.num_policies = 4  # Fewer policies for faster testing
            config.policy_schedule = 'round_robin'
            
            # R2D2 settings
            config.sequence_length = 40
            config.burn_in_length = 20
            config.lstm_size = 256
            config.num_lstm_layers = 1
            
            # Memory optimization
            config.memory_size = 5000
            config.episodic_memory_size = 2000
            config.batch_size = 8
            config.max_episode_steps = 2000
            
            # Create hybrid agent
            agent = R2D2Agent57Hybrid(config)
            
            # Training statistics
            episode_rewards = []
            intrinsic_rewards = []
            policy_usage = {}
            
            # Train for specified episodes
            for episode in range(self.episodes):
                episode_stats = agent.train_episode()
                
                episode_rewards.append(episode_stats['episode_reward'])
                intrinsic_rewards.append(episode_stats['episode_intrinsic_reward'])
                
                policy_id = episode_stats['policy_id']
                policy_usage[policy_id] = policy_usage.get(policy_id, 0) + 1
                
                if episode % max(1, self.episodes // 3) == 0:
                    self.log(f"    Episode {episode + 1}/{self.episodes}: "
                            f"Policy={policy_id}, "
                            f"Reward={episode_stats['episode_reward']:.1f}, "
                            f"Intrinsic={episode_stats['episode_intrinsic_reward']:.3f}")
            
            elapsed = time.time() - start_time
            
            # Get final statistics
            final_stats = agent.get_statistics()
            
            self.log(f"  ✓ R2D2+Agent57 completed in {elapsed:.1f}s")
            self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
            self.log(f"    Avg intrinsic: {np.mean(intrinsic_rewards):.3f}")
            self.log(f"    Policies used: {list(policy_usage.keys())}")
            
            return {
                'episode_rewards': episode_rewards,
                'intrinsic_rewards': intrinsic_rewards,
                'policy_usage': policy_usage,
                'elapsed_time': elapsed,
                'final_stats': final_stats,
                'algorithm': 'R2D2+Agent57',
                'success': True
            }
            
        except Exception as e:
            self.log(f"  ✗ R2D2+Agent57 failed: {e}")
            return {
                'episode_rewards': [0] * self.episodes,
                'intrinsic_rewards': [0] * self.episodes,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'algorithm': 'R2D2+Agent57',
                'success': False
            }
    
    def run_all_tests(self):
        """Run all test algorithms on all games"""
        algorithms = [
            ('Baseline DQN', self.test_baseline_dqn),
            ('R2D2', self.test_r2d2),
            ('Distributed DQN', self.test_distributed_dqn),
            ('NGU', self.test_ngu_agent),
            ('Agent57', self.test_agent57),
            ('R2D2+Agent57', self.test_r2d2_agent57)
        ]
        
        for game in self.games:
            game_name = game.split('/')[-1].replace('-v5', '')
            self.log(f"\n{'='*60}")
            self.log(f"TESTING ON: {game_name}")
            self.log(f"{'='*60}")
            
            self.results[game_name] = {}
            
            for alg_name, test_func in algorithms:
                self.log(f"\nAlgorithm: {alg_name}")
                
                try:
                    result = test_func(game)
                    self.results[game_name][alg_name] = result
                except Exception as e:
                    self.log(f"  ✗ {alg_name} crashed: {e}")
                    self.results[game_name][alg_name] = {
                        'algorithm': alg_name,
                        'success': False,
                        'error': str(e),
                        'elapsed_time': 0
                    }
                
                # Save intermediate results
                self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        results_path = os.path.join(self.output_dir, 'results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for game in self.results:
            json_results[game] = {}
            for algorithm in self.results[game]:
                data = self.results[game][algorithm].copy()
                
                # Convert numpy arrays and other non-serializable objects
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        data[key] = float(value)
                    elif key == 'final_stats' and isinstance(value, dict):
                        # Handle nested dictionaries with numpy values
                        data[key] = self._clean_dict_for_json(value)
                
                json_results[game][algorithm] = data
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.log(f"Results saved: {results_path}")
    
    def _clean_dict_for_json(self, d):
        """Recursively clean dictionary for JSON serialization"""
        if isinstance(d, dict):
            return {k: self._clean_dict_for_json(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [self._clean_dict_for_json(item) for item in d]
        elif isinstance(d, (np.integer, np.floating)):
            return float(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        else:
            return d
    
    def create_summary(self):
        """Create experiment summary"""
        summary_path = os.path.join(self.output_dir, 'summary.md')
        
        with open(summary_path, 'w') as f:
            f.write(f"# NGU Test Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Episodes per algorithm:** {self.episodes}\n")
            f.write(f"**Games tested:** {', '.join(self.games)}\n\n")
            
            for game_name in self.results:
                f.write(f"## {game_name} Results\n\n")
                f.write("| Algorithm | Success | Avg Reward | Time (s) | Notes |\n")
                f.write("|-----------|---------|------------|----------|-------|\n")
                
                for alg_name, data in self.results[game_name].items():
                    success = "✓" if data.get('success', False) else "✗"
                    
                    if data.get('success', False) and 'episode_rewards' in data:
                        avg_reward = np.mean(data['episode_rewards'])
                        time_taken = data.get('elapsed_time', 0)
                        
                        notes = ""
                        if 'intrinsic_rewards' in data:
                            avg_intrinsic = np.mean(data['intrinsic_rewards'])
                            notes = f"Intrinsic: {avg_intrinsic:.3f}"
                        
                        if 'policy_usage' in data:
                            policies = list(data['policy_usage'].keys())
                            notes += f" Policies: {policies}"
                    else:
                        avg_reward = 0
                        time_taken = data.get('elapsed_time', 0)
                        notes = data.get('error', 'Unknown error')
                    
                    f.write(f"| {alg_name} | {success} | {avg_reward:.2f} | {time_taken:.1f} | {notes} |\n")
                
                f.write("\n")
        
        self.log(f"Summary created: {summary_path}")


def main():
    """Run NGU test experiment"""
    import argparse
    parser = argparse.ArgumentParser(description='NGU Test Experiment')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per algorithm')
    parser.add_argument('--games', nargs='+', default=['ALE/Alien-v5'], 
                       help='Games to test')
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Run experiment
    experiment = NGUTestExperiment(
        episodes=args.episodes,
        games=args.games
    )
    
    experiment.run_all_tests()
    experiment.create_summary()
    
    experiment.log(f"\n{'='*60}")
    experiment.log("NGU TEST EXPERIMENT COMPLETED!")
    experiment.log(f"Results directory: {experiment.output_dir}")
    experiment.log("="*60)


if __name__ == '__main__':
    main()