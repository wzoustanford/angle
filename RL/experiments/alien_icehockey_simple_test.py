#!/usr/bin/env python3
"""
Simplified test of 3 single-threaded algorithms on Alien and Ice Hockey
(Basic DQN, DQN + Dueling, DQN + Priority)
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model import DQNAgent


class SimpleAlienIceHockeyTest:
    """Simple test with single-threaded algorithms only"""
    
    def __init__(self, episodes=2):
        self.episodes = episodes
        self.games = ['ALE/Alien-v5', 'ALE/IceHockey-v5']
        self.game_names = ['Alien', 'IceHockey']
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'./results/simple_test_{timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        print(f"Simple Test: {episodes} episodes per algorithm per game")
        print(f"Output: {self.output_dir}")
    
    def create_basic_config(self, env_name: str) -> AgentConfig:
        """Basic DQN"""
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
    
    def create_dueling_config(self, env_name: str) -> AgentConfig:
        """DQN + Dueling"""
        config = self.create_basic_config(env_name)
        config.use_dueling = True
        return config
    
    def create_priority_config(self, env_name: str) -> AgentConfig:
        """DQN + Priority"""
        config = self.create_basic_config(env_name)
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        return config
    
    def run_single_algorithm(self, name: str, config, game_name: str, max_steps=1000):
        """Run one algorithm on one game"""
        print(f"\n  {name} on {game_name}:")
        
        try:
            agent = DQNAgent(config)
            episode_rewards = []
            
            for ep in range(self.episodes):
                obs, _ = agent.env.reset()
                state = agent.frame_stack.reset(obs)
                agent.reset_hidden_state()
                episode_reward = 0
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
                    
                    # Update networks
                    if agent.steps_done % agent.config.policy_update_interval == 0:
                        agent.update_q_network()
                    
                    if agent.steps_done % agent.config.target_update_freq == 0:
                        agent.update_target_network()
                
                # Update epsilon
                agent.epsilon = max(agent.config.epsilon_end, 
                                  agent.epsilon * agent.config.epsilon_decay)
                
                episode_rewards.append(episode_reward)
                print(f"    Episode {ep+1}: reward={episode_reward:.1f}, steps={steps}")
            
            avg_reward = np.mean(episode_rewards)
            print(f"    Average: {avg_reward:.2f}")
            
            return episode_rewards
            
        except Exception as e:
            print(f"    Error: {e}")
            return [0] * self.episodes
    
    def run_all_tests(self):
        """Run all tests"""
        algorithms = [
            ('Basic DQN', lambda env: self.create_basic_config(env)),
            ('DQN + Dueling', lambda env: self.create_dueling_config(env)),
            ('DQN + Priority', lambda env: self.create_priority_config(env)),
        ]
        
        for game, game_name in zip(self.games, self.game_names):
            print(f"\n{'='*60}")
            print(f"GAME: {game_name}")
            print(f"{'='*60}")
            
            self.results[game_name] = {}
            
            for alg_name, config_fn in algorithms:
                config = config_fn(game)
                rewards = self.run_single_algorithm(alg_name, config, game_name)
                self.results[game_name][alg_name] = rewards
    
    def print_summary(self):
        """Print results summary"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        
        for game_name in self.game_names:
            print(f"\n{game_name}:")
            print("-"*30)
            
            for alg_name, rewards in self.results[game_name].items():
                avg = np.mean(rewards)
                print(f"  {alg_name:<20}: avg={avg:>8.2f}, rewards={rewards}")
    
    def create_plot(self):
        """Create simple comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Algorithm Comparison ({self.episodes} episodes)', fontsize=14)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, game_name in enumerate(self.game_names):
            ax = axes[i]
            
            alg_names = list(self.results[game_name].keys())
            avg_rewards = [np.mean(self.results[game_name][alg]) for alg in alg_names]
            
            bars = ax.bar(range(len(alg_names)), avg_rewards, color=colors[:len(alg_names)])
            ax.set_title(game_name)
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Average Reward')
            ax.set_xticks(range(len(alg_names)))
            ax.set_xticklabels(alg_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, avg_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple algorithm test')
    parser.add_argument('--episodes', type=int, default=2, help='Episodes per algorithm')
    args = parser.parse_args()
    
    try:
        test = SimpleAlienIceHockeyTest(episodes=args.episodes)
        test.run_all_tests()
        test.print_summary()
        test.create_plot()
        print("\n✓ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()