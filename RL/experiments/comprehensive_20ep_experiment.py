#!/usr/bin/env python3
"""
Comprehensive 20-Episode Experiment with All Algorithms

Tests all implemented RL algorithms including:
- Baseline DQN
- R2D2 (LSTM-based)
- Distributed DQN (parallel workers)
- NGU (intrinsic motivation)
- Agent57 (multi-policy)
- R2D2+Agent57 Hybrid

Also includes time-matched distributed experiment to show speedup benefits.
"""

import sys
import os
import time
import json
import gc
import torch
import numpy as np
import psutil
import traceback
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


class Comprehensive20EpExperiment:
    """Comprehensive 20-episode experiment with all algorithms"""
    
    def __init__(self, episodes: int = 20):
        self.episodes = episodes
        self.game = 'ALE/Alien-v5'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./experiments/results/comprehensive_20ep_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log(f"Comprehensive 20-Episode Experiment Started")
        self.log(f"Episodes per algorithm: {episodes}")
        self.log(f"Game: {self.game}")
        self.log(f"Output directory: {self.output_dir}")
        
        # Estimate runtime
        estimated_time = self.estimate_runtime()
        self.log(f"Estimated runtime: {estimated_time:.1f} minutes")
        self.log("=" * 60)
        
        # System info
        self.log_system_info()
    
    def estimate_runtime(self) -> float:
        """Estimate total runtime in minutes"""
        # Based on 4-episode benchmarks
        time_per_episode = {
            'baseline': 5.0,
            'prioritized': 5.2,
            'r2d2': 1.2,
            'distributed': 2.5,
            'ngu': 5.5,
            'r2d2_agent57': 4.5,
            'distributed_matched': 2.5
        }
        
        total_seconds = sum(t * self.episodes for t in time_per_episode.values())
        cleanup_overhead = 7 * 3  # 7 algorithms × 3 seconds cleanup
        total_seconds += cleanup_overhead
        
        return total_seconds / 60.0
    
    def log_system_info(self):
        """Log system information"""
        try:
            process = psutil.Process()
            mem_info = psutil.virtual_memory()
            
            self.log(f"System Memory: {mem_info.total/1024/1024/1024:.1f}GB total, "
                    f"{mem_info.available/1024/1024/1024:.1f}GB available")
            self.log(f"Process Memory: {process.memory_info().rss/1024/1024:.1f}MB")
            
            if torch.cuda.is_available():
                self.log(f"GPU: {torch.cuda.get_device_name(0)}")
                self.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB")
        except Exception as e:
            self.log(f"Could not get system info: {e}")
    
    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message, flush=True)  # Force flush for real-time output
        with open(self.log_file, 'a') as f:
            f.write(f"{full_message}\n")
            f.flush()  # Ensure written to disk immediately
    
    def cleanup_agent(self, agent):
        """Clean up agent and free memory"""
        try:
            # Close environment if it exists
            if hasattr(agent, 'env'):
                agent.env.close()
                del agent.env
            
            # Delete common attributes
            attrs_to_delete = [
                'replay_buffer', 'q_network', 'target_network', 
                'optimizer', 'frame_stack', 'network', 'predictor_network',
                'target_network', 'intrinsic_reward_module', 'meta_controller',
                'policy_lstms', 'episodic_memory'
            ]
            
            for attr in attrs_to_delete:
                if hasattr(agent, attr):
                    delattr(agent, attr)
            
            # Delete the agent itself
            del agent
        except Exception as e:
            self.log(f"  Warning during cleanup: {e}")
    
    def cleanup_memory(self):
        """Force memory cleanup between algorithms"""
        import psutil
        
        # Get memory before cleanup
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force Python garbage collection (multiple passes)
        for _ in range(3):
            gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        
        # Small delay to let system reclaim memory
        time.sleep(3)
        
        # Get memory after cleanup
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        freed = mem_before - mem_after
        
        self.log(f"  Memory cleaned: {mem_before:.1f}MB → {mem_after:.1f}MB (freed {freed:.1f}MB)")
    
    def run_baseline_dqn(self) -> Dict[str, Any]:
        """Run baseline DQN"""
        self.log(f"\n1. Baseline DQN")
        start_time = time.time()
        
        config = AgentConfig()
        config.env_name = self.game
        config.use_dueling = False
        config.use_prioritized_replay = False
        config.memory_size = 10000  # Reduced for memory
        config.min_replay_size = 2000
        config.batch_size = 32
        config.max_episode_steps = 3000
        
        agent = DQNAgent(config)
        episode_rewards = []
        
        for episode in range(self.episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done and episode_steps < config.max_episode_steps:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                next_state = agent.frame_stack.append(next_obs)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if agent.steps_done % 4 == 0:
                    agent.update_q_network()
                
                if agent.steps_done % 1000 == 0:
                    agent.update_target_network()
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                agent.steps_done += 1
            
            agent.epsilon = max(agent.config.epsilon_end, 
                              agent.epsilon * agent.config.epsilon_decay)
            
            episode_rewards.append(episode_reward)
            
            if episode % 5 == 0:
                self.log(f"  Episode {episode + 1}/{self.episodes}: Reward={episode_reward:.1f}")
        
        elapsed = time.time() - start_time
        self.log(f"  ✓ Completed in {elapsed:.1f}s, Avg reward: {np.mean(episode_rewards):.2f}")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'Baseline DQN',
            'episode_rewards': episode_rewards,
            'elapsed_time': elapsed,
            'avg_reward': np.mean(episode_rewards)
        }
    
    def run_dqn_prioritized(self) -> Dict[str, Any]:
        """Run DQN with prioritized replay for time comparison"""
        self.log(f"\n2. DQN + Prioritized Replay")
        start_time = time.time()
        
        config = AgentConfig()
        config.env_name = self.game
        config.use_dueling = False
        config.use_prioritized_replay = True
        config.memory_size = 10000  # Reduced for memory
        config.min_replay_size = 2000
        config.batch_size = 32
        config.max_episode_steps = 3000
        
        agent = DQNAgent(config)
        episode_rewards = []
        
        for episode in range(self.episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done and episode_steps < config.max_episode_steps:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                next_state = agent.frame_stack.append(next_obs)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if agent.steps_done % 4 == 0:
                    agent.update_q_network()
                
                if agent.steps_done % 1000 == 0:
                    agent.update_target_network()
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                agent.steps_done += 1
            
            agent.epsilon = max(agent.config.epsilon_end, 
                              agent.epsilon * agent.config.epsilon_decay)
            
            episode_rewards.append(episode_reward)
            
            if episode % 5 == 0:
                self.log(f"  Episode {episode + 1}/{self.episodes}: Reward={episode_reward:.1f}")
        
        elapsed = time.time() - start_time
        self.log(f"  ✓ Completed in {elapsed:.1f}s, Avg reward: {np.mean(episode_rewards):.2f}")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'DQN + Prioritized',
            'episode_rewards': episode_rewards,
            'elapsed_time': elapsed,
            'avg_reward': np.mean(episode_rewards)
        }
    
    def run_r2d2(self) -> Dict[str, Any]:
        """Run R2D2"""
        self.log(f"\n3. R2D2 (LSTM)")
        start_time = time.time()
        
        config = AgentConfig()
        config.env_name = self.game
        config.use_r2d2 = True
        config.use_dueling = True
        config.sequence_length = 80
        config.burn_in_length = 40
        config.lstm_size = 512
        config.memory_size = 10000  # Reduced for memory
        config.max_episode_steps = 3000
        
        agent = DQNAgent(config)
        episode_rewards = []
        
        for episode in range(self.episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            agent.reset_hidden_state()
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done and episode_steps < config.max_episode_steps:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                next_state = agent.frame_stack.append(next_obs)
                
                if hasattr(agent.replay_buffer, 'push_transition'):
                    agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                else:
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                
                if agent.steps_done % 4 == 0:
                    agent.update_q_network()
                
                if agent.steps_done % 1000 == 0:
                    agent.update_target_network()
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                agent.steps_done += 1
            
            agent.epsilon = max(agent.config.epsilon_end, 
                              agent.epsilon * agent.config.epsilon_decay)
            
            episode_rewards.append(episode_reward)
            
            if episode % 5 == 0:
                self.log(f"  Episode {episode + 1}/{self.episodes}: Reward={episode_reward:.1f}")
        
        elapsed = time.time() - start_time
        self.log(f"  ✓ Completed in {elapsed:.1f}s, Avg reward: {np.mean(episode_rewards):.2f}")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'R2D2',
            'episode_rewards': episode_rewards,
            'elapsed_time': elapsed,
            'avg_reward': np.mean(episode_rewards)
        }
    
    def run_distributed_dqn(self, num_workers: int = 4) -> Dict[str, Any]:
        """Run Distributed DQN"""
        self.log(f"\n4. Distributed DQN ({num_workers} workers)")
        start_time = time.time()
        
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = num_workers
        config.use_prioritized_replay = True
        config.memory_size = 10000  # Reduced for memory
        config.batch_size = 32
        
        agent = DistributedDQNAgent(config, num_workers=num_workers)
        results = agent.train_distributed(total_episodes=self.episodes)
        
        elapsed = time.time() - start_time
        
        env_stats = results.get('env_stats', {})
        avg_reward = env_stats.get('overall_avg_reward', 0)
        total_episodes = env_stats.get('total_episodes', 0)
        
        self.log(f"  ✓ Completed in {elapsed:.1f}s")
        self.log(f"    Total episodes: {total_episodes}")
        self.log(f"    Avg reward: {avg_reward:.2f}")
        self.log(f"    Throughput: {total_episodes/elapsed:.1f} episodes/sec")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'Distributed DQN',
            'episode_rewards': [avg_reward] * self.episodes,
            'elapsed_time': elapsed,
            'avg_reward': avg_reward,
            'total_episodes': total_episodes,
            'throughput': total_episodes/elapsed,
            'num_workers': num_workers
        }
    
    def run_distributed_time_matched(self, target_time: float) -> Dict[str, Any]:
        """Run Distributed DQN for same time as DQN+Prioritized"""
        self.log(f"\n7. Distributed DQN (Time-matched: {target_time:.1f}s)")
        start_time = time.time()
        
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = 4
        config.use_prioritized_replay = True
        config.memory_size = 10000  # Reduced for memory
        config.batch_size = 32
        
        agent = DistributedDQNAgent(config, num_workers=4)
        
        # Run episodes until time limit
        total_episodes = 0
        all_rewards = []
        
        while (time.time() - start_time) < target_time:
            remaining_time = target_time - (time.time() - start_time)
            if remaining_time < 5:  # Stop if less than 5 seconds left
                break
            
            # Run batch of episodes
            batch_size = min(10, max(2, int(remaining_time / 10)))
            results = agent.train_distributed(total_episodes=batch_size)
            
            env_stats = results.get('env_stats', {})
            batch_episodes = env_stats.get('total_episodes', 0)
            batch_reward = env_stats.get('overall_avg_reward', 0)
            
            total_episodes += batch_episodes
            all_rewards.extend([batch_reward] * batch_episodes)
            
            self.log(f"    Batch: {batch_episodes} episodes, avg reward: {batch_reward:.1f}")
        
        elapsed = time.time() - start_time
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        
        self.log(f"  ✓ Completed in {elapsed:.1f}s")
        self.log(f"    Total episodes: {total_episodes}")
        self.log(f"    Avg reward: {avg_reward:.2f}")
        self.log(f"    Throughput: {total_episodes/elapsed:.1f} episodes/sec")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'Distributed DQN (Time-matched)',
            'episode_rewards': all_rewards,
            'elapsed_time': elapsed,
            'avg_reward': avg_reward,
            'total_episodes': total_episodes,
            'throughput': total_episodes/elapsed,
            'num_workers': 4
        }
    
    def run_ngu(self) -> Dict[str, Any]:
        """Run NGU"""
        self.log(f"\n5. NGU (Never Give Up)")
        start_time = time.time()
        
        config = create_ngu_config_for_game(self.game, episodes=self.episodes, use_agent57=False)
        config.memory_size = 10000  # Reduced for memory
        config.episodic_memory_size = 5000  # Reduced for memory
        
        agent = NGUAgent(config)
        episode_rewards = []
        intrinsic_rewards = []
        
        for episode in range(self.episodes):
            episode_stats = agent.train_episode()
            episode_rewards.append(episode_stats['episode_reward'])
            intrinsic_rewards.append(episode_stats['episode_intrinsic_reward'])
            
            if episode % 5 == 0:
                self.log(f"  Episode {episode + 1}/{self.episodes}: "
                        f"Reward={episode_stats['episode_reward']:.1f}, "
                        f"Intrinsic={episode_stats['episode_intrinsic_reward']:.1f}")
        
        elapsed = time.time() - start_time
        self.log(f"  ✓ Completed in {elapsed:.1f}s")
        self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
        self.log(f"    Avg intrinsic: {np.mean(intrinsic_rewards):.1f}")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'NGU',
            'episode_rewards': episode_rewards,
            'intrinsic_rewards': intrinsic_rewards,
            'elapsed_time': elapsed,
            'avg_reward': np.mean(episode_rewards),
            'avg_intrinsic': np.mean(intrinsic_rewards)
        }
    
    def run_r2d2_agent57(self) -> Dict[str, Any]:
        """Run R2D2+Agent57 Hybrid"""
        self.log(f"\n6. R2D2+Agent57 Hybrid")
        start_time = time.time()
        
        config = Agent57Config()
        config.env_name = self.game
        config.num_policies = 8
        config.memory_size = 10000  # Reduced for memory
        config.episodic_memory_size = 5000  # Reduced for memory
        config.sequence_length = 80
        config.burn_in_length = 40
        config.max_episode_steps = 3000
        
        agent = R2D2Agent57Hybrid(config)
        episode_rewards = []
        intrinsic_rewards = []
        policy_usage = {}
        
        for episode in range(self.episodes):
            episode_stats = agent.train_episode()
            episode_rewards.append(episode_stats['episode_reward'])
            intrinsic_rewards.append(episode_stats['episode_intrinsic_reward'])
            
            policy_id = episode_stats['policy_id']
            policy_usage[policy_id] = policy_usage.get(policy_id, 0) + 1
            
            if episode % 5 == 0:
                self.log(f"  Episode {episode + 1}/{self.episodes}: "
                        f"Policy={policy_id}, "
                        f"Reward={episode_stats['episode_reward']:.1f}")
        
        elapsed = time.time() - start_time
        self.log(f"  ✓ Completed in {elapsed:.1f}s")
        self.log(f"    Avg reward: {np.mean(episode_rewards):.2f}")
        self.log(f"    Policies used: {list(policy_usage.keys())}")
        
        # Clean up memory
        self.cleanup_agent(agent)
        self.cleanup_memory()
        
        return {
            'algorithm': 'R2D2+Agent57',
            'episode_rewards': episode_rewards,
            'intrinsic_rewards': intrinsic_rewards,
            'elapsed_time': elapsed,
            'avg_reward': np.mean(episode_rewards),
            'avg_intrinsic': np.mean(intrinsic_rewards),
            'policy_usage': policy_usage
        }
    
    def run_with_error_handling(self, algorithm_name: str, run_func, *args, **kwargs):
        """Run algorithm with error handling and logging"""
        try:
            self.log(f"\nStarting {algorithm_name}...")
            result = run_func(*args, **kwargs)
            self.log(f"✓ {algorithm_name} completed successfully")
            return result
        except Exception as e:
            self.log(f"✗ {algorithm_name} failed: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            return {
                'algorithm': algorithm_name,
                'error': str(e),
                'episode_rewards': [],
                'elapsed_time': 0,
                'avg_reward': 0
            }
    
    def run_experiment(self):
        """Run the complete experiment"""
        self.log("\n" + "="*60)
        self.log("STARTING COMPREHENSIVE 20-EPISODE EXPERIMENT")
        self.log("="*60)
        
        total_start_time = time.time()
        
        # Run all algorithms with error handling
        self.results['baseline_dqn'] = self.run_with_error_handling(
            "Baseline DQN", self.run_baseline_dqn)
        
        self.results['dqn_prioritized'] = self.run_with_error_handling(
            "DQN + Prioritized", self.run_dqn_prioritized)
        
        self.results['r2d2'] = self.run_with_error_handling(
            "R2D2", self.run_r2d2)
        
        self.results['distributed_dqn'] = self.run_with_error_handling(
            "Distributed DQN", self.run_distributed_dqn, num_workers=4)
        
        self.results['ngu'] = self.run_with_error_handling(
            "NGU", self.run_ngu)
        
        self.results['r2d2_agent57'] = self.run_with_error_handling(
            "R2D2+Agent57", self.run_r2d2_agent57)
        
        # Run time-matched distributed experiment
        if 'elapsed_time' in self.results.get('dqn_prioritized', {}):
            prioritized_time = self.results['dqn_prioritized']['elapsed_time']
            self.results['distributed_time_matched'] = self.run_with_error_handling(
                "Distributed Time-matched", self.run_distributed_time_matched, prioritized_time)
        
        # Save results after each algorithm in case of crash
        self.save_results()
        self.create_summary()
        
        total_elapsed = time.time() - total_start_time
        self.log("\n" + "="*60)
        self.log("EXPERIMENT COMPLETED!")
        self.log(f"Total runtime: {total_elapsed/60:.1f} minutes")
        self.log(f"Results saved to: {self.output_dir}")
        self.log("="*60)
    
    def save_results(self):
        """Save results to JSON"""
        results_path = os.path.join(self.output_dir, 'results.json')
        
        # Convert numpy arrays to lists for JSON
        json_results = {}
        for key, data in self.results.items():
            json_results[key] = {}
            for field, value in data.items():
                if isinstance(value, np.ndarray):
                    json_results[key][field] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key][field] = float(value)
                else:
                    json_results[key][field] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def create_summary(self):
        """Create markdown summary"""
        summary_path = os.path.join(self.output_dir, 'summary.md')
        
        with open(summary_path, 'w') as f:
            f.write("# Comprehensive 20-Episode Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Episodes:** {self.episodes}\n")
            f.write(f"**Game:** Alien\n\n")
            
            f.write("## Performance Comparison\n\n")
            f.write("| Algorithm | Avg Reward | Time (s) | Episodes | Throughput | Notes |\n")
            f.write("|-----------|------------|----------|----------|------------|-------|\n")
            
            # Sort by average reward
            sorted_results = sorted(self.results.items(), 
                                  key=lambda x: x[1]['avg_reward'], 
                                  reverse=True)
            
            for key, data in sorted_results:
                alg = data['algorithm']
                avg_reward = data['avg_reward']
                elapsed = data['elapsed_time']
                
                if 'total_episodes' in data:
                    episodes = data['total_episodes']
                    throughput = f"{data['throughput']:.2f} ep/s"
                else:
                    episodes = self.episodes
                    throughput = f"{episodes/elapsed:.2f} ep/s"
                
                notes = ""
                if 'avg_intrinsic' in data:
                    notes = f"Intrinsic: {data['avg_intrinsic']:.1f}"
                if 'num_workers' in data:
                    notes += f" Workers: {data['num_workers']}"
                
                f.write(f"| {alg} | {avg_reward:.2f} | {elapsed:.1f} | {episodes} | {throughput} | {notes} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find best performer
            best = sorted_results[0]
            f.write(f"- **Best Performance:** {best[1]['algorithm']} with {best[1]['avg_reward']:.2f} average reward\n")
            
            # Compare distributed speedup
            if 'distributed_time_matched' in self.results:
                dist_episodes = self.results['distributed_time_matched']['total_episodes']
                priority_episodes = self.episodes
                speedup = dist_episodes / priority_episodes
                f.write(f"- **Distributed Speedup:** {speedup:.1f}x ({dist_episodes} vs {priority_episodes} episodes)\n")
            
            # Exploration algorithms
            if 'ngu' in self.results:
                ngu_intrinsic = self.results['ngu'].get('avg_intrinsic', 0)
                f.write(f"- **NGU Exploration:** Average intrinsic reward of {ngu_intrinsic:.1f}\n")
            
            f.write("\n## Episode Rewards Over Time\n\n")
            f.write("```\n")
            for key, data in self.results.items():
                if 'episode_rewards' in data and len(data['episode_rewards']) == self.episodes:
                    rewards = data['episode_rewards']
                    f.write(f"{data['algorithm']:25} | ")
                    # Show reward at episodes 5, 10, 15, 20
                    for i in [4, 9, 14, 19]:
                        if i < len(rewards):
                            f.write(f"{rewards[i]:6.0f} ")
                    f.write(f"| Final: {rewards[-1]:.0f}\n")
            f.write("```\n")


def main():
    """Run comprehensive experiment"""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive 20-Episode Experiment')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per algorithm')
    args = parser.parse_args()
    
    experiment = Comprehensive20EpExperiment(episodes=args.episodes)
    experiment.run_experiment()


if __name__ == '__main__':
    main()