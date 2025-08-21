#!/usr/bin/env python3
"""
Sequential 20-Episode Experiment - Runs one algorithm at a time to avoid memory issues
"""

import sys
import os
import time
import json
import gc
import torch
import numpy as np
import psutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import create_ngu_config_for_game, Agent57Config
from model.ngu_agent import NGUAgent
from model.dqn_agent import DQNAgent
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid
from config.AgentConfig import AgentConfig


class Sequential20EpExperiment:
    """Run algorithms sequentially with aggressive memory management"""
    
    def __init__(self):
        self.episodes = 20
        self.game = 'ALE/Alien-v5'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./experiments/results/sequential_20ep_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        self.current_results = {}
        
        self.log(f"Sequential 20-Episode Experiment")
        self.log(f"Episodes: {self.episodes}")
        self.log(f"Game: {self.game}")
        self.log(f"Output: {self.output_dir}")
        self.log_memory_status()
        self.log("=" * 60)
    
    def log(self, message: str):
        """Log with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message, flush=True)
        with open(self.log_file, 'a') as f:
            f.write(f"{full_message}\n")
            f.flush()
    
    def log_memory_status(self):
        """Log current memory usage"""
        process = psutil.Process()
        mem_info = psutil.virtual_memory()
        self.log(f"Memory: {process.memory_info().rss/1024/1024:.1f}MB process, "
                f"{mem_info.available/1024/1024/1024:.1f}GB available")
    
    def aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Multiple GC passes
        for _ in range(5):
            gc.collect()
        
        # Clear GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Wait for system
        time.sleep(5)
        
        self.log_memory_status()
    
    def run_single_algorithm(self, name: str, config_func, agent_class):
        """Run a single algorithm and immediately clean up"""
        self.log(f"\n{'='*60}")
        self.log(f"Running: {name}")
        self.log_memory_status()
        
        try:
            start_time = time.time()
            
            # Create config and agent
            config = config_func()
            agent = agent_class(config)
            
            episode_rewards = []
            
            # Run episodes
            for ep in range(self.episodes):
                if hasattr(agent, 'train_episode'):
                    # NGU/Agent57 style
                    stats = agent.train_episode()
                    reward = stats['episode_reward']
                else:
                    # DQN style
                    obs, _ = agent.env.reset()
                    state = agent.frame_stack.reset(obs)
                    if hasattr(agent, 'reset_hidden_state'):
                        agent.reset_hidden_state()
                    
                    episode_reward = 0
                    done = False
                    steps = 0
                    max_steps = 3000
                    
                    while not done and steps < max_steps:
                        action = agent.select_action(state)
                        next_obs, reward_step, terminated, truncated, _ = agent.env.step(action)
                        done = terminated or truncated
                        
                        next_state = agent.frame_stack.append(next_obs)
                        
                        # Store transition
                        if hasattr(agent.replay_buffer, 'push_transition'):
                            agent.replay_buffer.push_transition(state, action, reward_step, next_state, done)
                        else:
                            agent.replay_buffer.push(state, action, reward_step, next_state, done)
                        
                        # Update
                        if agent.steps_done % 4 == 0:
                            agent.update_q_network()
                        if agent.steps_done % 1000 == 0:
                            agent.update_target_network()
                        
                        state = next_state
                        episode_reward += reward_step
                        steps += 1
                        agent.steps_done += 1
                    
                    agent.epsilon = max(0.01, agent.epsilon * 0.995)
                    reward = episode_reward
                
                episode_rewards.append(reward)
                
                if (ep + 1) % 5 == 0:
                    self.log(f"  Episode {ep+1}/{self.episodes}: Reward={reward:.1f}")
            
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards)
            
            self.log(f"✓ Completed in {elapsed:.1f}s")
            self.log(f"  Average reward: {avg_reward:.2f}")
            
            result = {
                'algorithm': name,
                'episode_rewards': episode_rewards,
                'elapsed_time': elapsed,
                'avg_reward': avg_reward
            }
            
            # Clean up agent completely
            if hasattr(agent, 'env'):
                agent.env.close()
            del agent
            
            # Save result immediately
            self.current_results[name] = result
            self.save_results()
            
            return result
            
        except Exception as e:
            self.log(f"✗ Failed: {str(e)}")
            return {
                'algorithm': name,
                'error': str(e),
                'episode_rewards': [],
                'avg_reward': 0
            }
        finally:
            # Always cleanup
            self.aggressive_cleanup()
    
    def save_results(self):
        """Save current results"""
        path = os.path.join(self.output_dir, 'results.json')
        with open(path, 'w') as f:
            json.dump(self.current_results, f, indent=2)
        self.log(f"Results saved to {path}")
    
    def create_summary(self):
        """Create markdown summary"""
        path = os.path.join(self.output_dir, 'summary.md')
        
        with open(path, 'w') as f:
            f.write("# Sequential 20-Episode Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Episodes:** {self.episodes}\n\n")
            
            f.write("## Results\n\n")
            f.write("| Algorithm | Avg Reward | Time (s) | Status |\n")
            f.write("|-----------|------------|----------|--------|\n")
            
            for name, data in self.current_results.items():
                if 'error' in data:
                    f.write(f"| {name} | - | - | Failed |\n")
                else:
                    f.write(f"| {name} | {data['avg_reward']:.2f} | {data['elapsed_time']:.1f} | ✓ |\n")
    
    def run_experiment(self):
        """Run all algorithms sequentially"""
        self.log("\nSTARTING SEQUENTIAL EXPERIMENT")
        self.log("=" * 60)
        
        total_start = time.time()
        
        # 1. Baseline DQN
        def dqn_config():
            config = AgentConfig()
            config.env_name = self.game
            config.memory_size = 5000  # Very small buffer
            config.batch_size = 16
            config.max_episode_steps = 3000
            return config
        
        self.run_single_algorithm("Baseline DQN", dqn_config, DQNAgent)
        
        # 2. R2D2
        def r2d2_config():
            config = AgentConfig()
            config.env_name = self.game
            config.use_r2d2 = True
            config.memory_size = 5000
            config.sequence_length = 40
            config.burn_in_length = 20
            config.lstm_size = 256
            config.max_episode_steps = 3000
            return config
        
        self.run_single_algorithm("R2D2", r2d2_config, DQNAgent)
        
        # 3. NGU
        def ngu_config():
            config = create_ngu_config_for_game(self.game, episodes=20, use_agent57=False)
            config.memory_size = 5000
            config.episodic_memory_size = 2000
            config.max_episode_steps = 3000
            return config
        
        self.run_single_algorithm("NGU", ngu_config, NGUAgent)
        
        # 4. R2D2+Agent57
        def agent57_config():
            config = Agent57Config()
            config.env_name = self.game
            config.num_policies = 4
            config.memory_size = 5000
            config.episodic_memory_size = 2000
            config.max_episode_steps = 3000
            return config
        
        self.run_single_algorithm("R2D2+Agent57", agent57_config, R2D2Agent57Hybrid)
        
        # Final summary
        total_elapsed = time.time() - total_start
        self.create_summary()
        
        self.log("\n" + "=" * 60)
        self.log("EXPERIMENT COMPLETED!")
        self.log(f"Total time: {total_elapsed/60:.1f} minutes")
        self.log(f"Results: {self.output_dir}")
        self.log("=" * 60)


if __name__ == '__main__':
    experiment = Sequential20EpExperiment()
    experiment.run_experiment()