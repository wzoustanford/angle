#!/usr/bin/env python3
"""
Autonomous 10-Episode Experiment
Reduced memory footprint for reliable completion
"""

import sys
import os
import time
import json
import numpy as np
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import create_ngu_config_for_game
from model.ngu_agent import NGUAgent
from model.dqn_agent import DQNAgent
from model.distributed_dqn_agent import DistributedDQNAgent
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid
from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig


class Auto10EpExperiment:
    def __init__(self):
        self.episodes = 10
        self.game = 'ALE/Alien-v5'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./experiments/results/auto_10ep_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}
        self.log_file = os.path.join(self.output_dir, 'experiment.log')
        
        self.log(f"Autonomous 10-Episode Experiment")
        self.log(f"Game: {self.game}")
        self.log("=" * 60)
    
    def log(self, message: str):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    def run_with_cleanup(self, name: str, run_func):
        """Run algorithm with automatic cleanup"""
        self.log(f"\n{name}")
        result = run_func()
        
        # Force garbage collection after each algorithm
        gc.collect()
        time.sleep(2)  # Give system time to reclaim memory
        
        return result
    
    def run_dqn(self):
        start_time = time.time()
        config = AgentConfig()
        config.env_name = self.game
        config.memory_size = 5000  # Small buffer
        config.batch_size = 16
        config.max_episode_steps = 1500  # Shorter episodes
        
        agent = DQNAgent(config)
        rewards = []
        
        for ep in range(self.episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            episode_reward = 0
            steps = 0
            
            done = False
            while not done and steps < config.max_episode_steps:
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
                steps += 1
                agent.steps_done += 1
            
            agent.epsilon *= agent.config.epsilon_decay
            rewards.append(episode_reward)
            if ep % 3 == 0:
                self.log(f"  Ep {ep+1}: {episode_reward:.0f}")
        
        elapsed = time.time() - start_time
        avg = np.mean(rewards)
        self.log(f"  ✓ Done in {elapsed:.1f}s, Avg: {avg:.1f}")
        
        # Clean up
        del agent
        
        return {'algorithm': 'DQN', 'rewards': rewards, 'time': elapsed, 'avg': avg}
    
    def run_r2d2(self):
        start_time = time.time()
        config = AgentConfig()
        config.env_name = self.game
        config.use_r2d2 = True
        config.sequence_length = 40
        config.burn_in_length = 20
        config.lstm_size = 256
        config.memory_size = 5000
        config.max_episode_steps = 1500
        
        agent = DQNAgent(config)
        rewards = []
        
        for ep in range(self.episodes):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            agent.reset_hidden_state()
            episode_reward = 0
            steps = 0
            
            done = False
            while not done and steps < config.max_episode_steps:
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
                if agent.steps_done % 500 == 0:
                    agent.update_target_network()
                
                state = next_state
                episode_reward += reward
                steps += 1
                agent.steps_done += 1
            
            agent.epsilon *= agent.config.epsilon_decay
            rewards.append(episode_reward)
            if ep % 3 == 0:
                self.log(f"  Ep {ep+1}: {episode_reward:.0f}")
        
        elapsed = time.time() - start_time
        avg = np.mean(rewards)
        self.log(f"  ✓ Done in {elapsed:.1f}s, Avg: {avg:.1f}")
        
        del agent
        
        return {'algorithm': 'R2D2', 'rewards': rewards, 'time': elapsed, 'avg': avg}
    
    def run_distributed(self):
        start_time = time.time()
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = 2  # Just 2 workers
        config.memory_size = 5000
        
        agent = DistributedDQNAgent(config, num_workers=2)
        results = agent.train_distributed(total_episodes=self.episodes)
        
        elapsed = time.time() - start_time
        avg = results['env_stats']['overall_avg_reward']
        total_ep = results['env_stats']['total_episodes']
        
        self.log(f"  ✓ Done in {elapsed:.1f}s")
        self.log(f"    Episodes: {total_ep}, Avg: {avg:.1f}")
        self.log(f"    Throughput: {total_ep/elapsed:.1f} ep/s")
        
        del agent
        
        return {'algorithm': 'Distributed', 'avg': avg, 'time': elapsed, 
                'episodes': total_ep, 'throughput': total_ep/elapsed}
    
    def run_ngu(self):
        start_time = time.time()
        config = create_ngu_config_for_game(self.game, episodes=10)
        config.memory_size = 5000
        config.episodic_memory_size = 2000
        config.max_episode_steps = 1500
        
        agent = NGUAgent(config)
        rewards = []
        intrinsic = []
        
        for ep in range(self.episodes):
            stats = agent.train_episode()
            rewards.append(stats['episode_reward'])
            intrinsic.append(stats['episode_intrinsic_reward'])
            if ep % 3 == 0:
                self.log(f"  Ep {ep+1}: {stats['episode_reward']:.0f}, Int: {stats['episode_intrinsic_reward']:.0f}")
        
        elapsed = time.time() - start_time
        avg = np.mean(rewards)
        avg_int = np.mean(intrinsic)
        self.log(f"  ✓ Done in {elapsed:.1f}s, Avg: {avg:.1f}, Intrinsic: {avg_int:.1f}")
        
        del agent
        
        return {'algorithm': 'NGU', 'rewards': rewards, 'time': elapsed, 
                'avg': avg, 'intrinsic': intrinsic, 'avg_intrinsic': avg_int}
    
    def run_r2d2_agent57(self):
        start_time = time.time()
        from config.NGUConfig import Agent57Config
        config = Agent57Config()
        config.env_name = self.game
        config.num_policies = 4  # Few policies
        config.memory_size = 5000
        config.episodic_memory_size = 2000
        config.max_episode_steps = 1500
        
        agent = R2D2Agent57Hybrid(config)
        rewards = []
        policies = []
        
        for ep in range(self.episodes):
            stats = agent.train_episode()
            rewards.append(stats['episode_reward'])
            policies.append(stats['policy_id'])
            if ep % 3 == 0:
                self.log(f"  Ep {ep+1}: {stats['episode_reward']:.0f}, Policy: {stats['policy_id']}")
        
        elapsed = time.time() - start_time
        avg = np.mean(rewards)
        self.log(f"  ✓ Done in {elapsed:.1f}s, Avg: {avg:.1f}")
        self.log(f"    Policies used: {set(policies)}")
        
        del agent
        
        return {'algorithm': 'R2D2+Agent57', 'rewards': rewards, 'time': elapsed,
                'avg': avg, 'policies': policies}
    
    def run_experiment(self):
        """Run complete experiment"""
        self.log("\nSTARTING AUTONOMOUS EXPERIMENT")
        self.log("="*60)
        
        # Run algorithms with cleanup
        self.results['dqn'] = self.run_with_cleanup("1. DQN", self.run_dqn)
        self.results['r2d2'] = self.run_with_cleanup("2. R2D2", self.run_r2d2)
        self.results['distributed'] = self.run_with_cleanup("3. Distributed DQN", self.run_distributed)
        self.results['ngu'] = self.run_with_cleanup("4. NGU", self.run_ngu)
        self.results['r2d2_agent57'] = self.run_with_cleanup("5. R2D2+Agent57", self.run_r2d2_agent57)
        
        # Distributed time-matched
        self.log("\n6. Distributed (Time-matched)")
        dqn_time = self.results['dqn']['time']
        start = time.time()
        
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = 4
        config.memory_size = 5000
        
        agent = DistributedDQNAgent(config, num_workers=4)
        total_ep = 0
        all_rewards = []
        
        while (time.time() - start) < dqn_time:
            if (dqn_time - (time.time() - start)) < 3:
                break
            batch = min(5, int((dqn_time - (time.time() - start)) / 5))
            res = agent.train_distributed(total_episodes=batch)
            total_ep += res['env_stats']['total_episodes']
            all_rewards.append(res['env_stats']['overall_avg_reward'])
        
        elapsed = time.time() - start
        avg = np.mean(all_rewards) if all_rewards else 0
        self.log(f"  ✓ {total_ep} episodes in {elapsed:.1f}s")
        self.log(f"    Speedup: {total_ep/self.episodes:.1f}x")
        
        self.results['distributed_matched'] = {
            'algorithm': 'Distributed (Matched)',
            'episodes': total_ep,
            'time': elapsed,
            'avg': avg,
            'speedup': total_ep/self.episodes
        }
        
        del agent
        gc.collect()
        
        # Save results
        self.save_results()
        self.create_summary()
        
        self.log("\n" + "="*60)
        self.log("EXPERIMENT COMPLETED!")
        self.log(f"Results: {self.output_dir}/summary.md")
    
    def save_results(self):
        """Save results as JSON"""
        path = os.path.join(self.output_dir, 'results.json')
        
        # Convert numpy to lists
        json_results = {}
        for key, data in self.results.items():
            json_results[key] = {}
            for field, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        json_results[key][field] = value.tolist()
                    else:
                        json_results[key][field] = value
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key][field] = float(value)
                else:
                    json_results[key][field] = value
        
        with open(path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def create_summary(self):
        """Create markdown summary"""
        path = os.path.join(self.output_dir, 'summary.md')
        
        with open(path, 'w') as f:
            f.write("# Autonomous 10-Episode Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Algorithm | Avg Reward | Time (s) | Episodes | Notes |\n")
            f.write("|-----------|------------|----------|----------|-------|\n")
            
            # Sort by average reward
            sorted_res = sorted(
                [(k, v) for k, v in self.results.items() if 'avg' in v],
                key=lambda x: x[1]['avg'],
                reverse=True
            )
            
            for key, data in sorted_res:
                alg = data['algorithm']
                avg = data.get('avg', 0)
                time_s = data.get('time', 0)
                episodes = data.get('episodes', self.episodes)
                
                notes = ""
                if 'avg_intrinsic' in data:
                    notes = f"Intrinsic: {data['avg_intrinsic']:.0f}"
                if 'throughput' in data:
                    notes = f"{data['throughput']:.1f} ep/s"
                if 'speedup' in data:
                    notes = f"{data['speedup']:.1f}x speedup"
                
                f.write(f"| {alg} | {avg:.1f} | {time_s:.1f} | {episodes} | {notes} |\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            best = sorted_res[0] if sorted_res else None
            if best:
                f.write(f"- **Best Performance:** {best[1]['algorithm']} ({best[1]['avg']:.1f} avg reward)\n")
            
            if 'distributed_matched' in self.results:
                speedup = self.results['distributed_matched']['speedup']
                f.write(f"- **Distributed Speedup:** {speedup:.1f}x in same time\n")
            
            if 'ngu' in self.results:
                intrinsic = self.results['ngu'].get('avg_intrinsic', 0)
                f.write(f"- **NGU Exploration:** {intrinsic:.0f} avg intrinsic reward\n")
            
            if 'r2d2_agent57' in self.results:
                policies = set(self.results['r2d2_agent57'].get('policies', []))
                f.write(f"- **R2D2+Agent57:** Used {len(policies)} different policies\n")


if __name__ == '__main__':
    experiment = Auto10EpExperiment()
    experiment.run_experiment()