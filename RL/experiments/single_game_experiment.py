#!/usr/bin/env python3
"""
Single Game Focused Experiment

Tests all four algorithms on one game with good episode count
to see clear learning curves and reward progression.
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


class SingleGameExperiment:
    """
    Focused experiment on a single game to show learning curves
    """
    
    def __init__(self, game='ALE/Breakout-v5', episodes=100, output_dir='./results/single_game'):
        self.game = game
        self.game_name = game.split('/')[-1].replace('-v5', '')
        self.episodes = episodes
        
        # Create unique output directory with game name and episode count
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"{self.game_name.lower()}_{episodes}ep_{timestamp}")
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Single Game Experiment: {self.game_name}")
        print(f"Episodes per algorithm: {self.episodes}")
        print(f"Output directory: {self.output_dir}")
    
    def create_basic_dqn_config(self) -> AgentConfig:
        """Basic DQN configuration"""
        config = AgentConfig()
        config.env_name = self.game
        config.use_r2d2 = False
        config.use_prioritized_replay = False
        
        # Optimized for faster training
        config.memory_size = 20000
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.target_update_freq = 500  # More frequent updates
        config.min_replay_size = 500    # Start training sooner
        config.save_interval = 50000    # Disable saving
        
        return config
    
    def create_prioritized_dqn_config(self) -> AgentConfig:
        """DQN + Prioritized Replay configuration"""
        config = self.create_basic_dqn_config()
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.priority_alpha = 0.6
        config.priority_beta_start = 0.4
        config.priority_beta_end = 1.0
        return config
    
    def create_r2d2_config(self) -> AgentConfig:
        """R2D2 configuration (single-threaded for now)"""
        config = self.create_prioritized_dqn_config()
        config.use_r2d2 = True
        config.lstm_size = 256          # Smaller for faster training
        config.sequence_length = 40     # Shorter sequences
        config.burn_in_length = 20
        return config
    
    def run_algorithm(self, name: str, config) -> tuple:
        """Run a single algorithm and return results"""
        print(f"\n{'='*50}")
        print(f"Running {name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            agent = DQNAgent(config)
            episode_rewards, losses = agent.train(num_episodes=self.episodes)
            
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards)
            final_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else avg_reward
            
            print(f"✓ {name} completed in {elapsed:.1f}s")
            print(f"  Episodes: {len(episode_rewards)}")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Final 10 episodes: {final_avg:.2f}")
            print(f"  Best episode: {max(episode_rewards):.2f}")
            
            return episode_rewards, losses, elapsed
            
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0
    
    def run_all_algorithms(self):
        """Run all algorithms"""
        algorithms = [
            ('Basic DQN', self.create_basic_dqn_config()),
            ('DQN + Priority', self.create_prioritized_dqn_config()),
            ('R2D2', self.create_r2d2_config()),
        ]
        
        print(f"Starting {self.game_name} experiment with {len(algorithms)} algorithms")
        
        for name, config in algorithms:
            rewards, losses, elapsed = self.run_algorithm(name, config)
            
            self.results[name] = {
                'rewards': rewards,
                'losses': losses,
                'elapsed': elapsed
            }
    
    def create_learning_curves(self):
        """Create detailed learning curves"""
        print("\nCreating learning curves...")
        
        # Colors for algorithms
        colors = {
            'Basic DQN': '#1f77b4',           # Blue
            'DQN + Priority': '#ff7f0e',      # Orange
            'R2D2': '#d62728'                 # Red
        }
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.game_name} - Algorithm Learning Comparison ({self.episodes} episodes)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Raw episode rewards
        ax1 = axes[0, 0]
        for name, color in colors.items():
            if name in self.results and self.results[name]['rewards']:
                rewards = self.results[name]['rewards']
                episodes = range(1, len(rewards) + 1)
                ax1.plot(episodes, rewards, color=color, alpha=0.7, linewidth=1, label=name)
        
        ax1.set_title('Raw Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Moving average (smoother trends)
        ax2 = axes[0, 1]
        window = max(5, self.episodes // 20)  # Adaptive window size
        
        for name, color in colors.items():
            if name in self.results and self.results[name]['rewards']:
                rewards = self.results[name]['rewards']
                if len(rewards) >= window:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    episodes = range(window, len(rewards) + 1)
                    ax2.plot(episodes, moving_avg, color=color, linewidth=2, label=name)
        
        ax2.set_title(f'Moving Average ({window}-episode window)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Training loss
        ax3 = axes[1, 0]
        for name, color in colors.items():
            if name in self.results and self.results[name]['losses']:
                losses = self.results[name]['losses']
                # Plot moving average of losses for clarity
                if len(losses) > 10:
                    loss_window = min(50, len(losses) // 10)
                    moving_loss = np.convolve(losses, np.ones(loss_window)/loss_window, mode='valid')
                    ax3.plot(range(loss_window, len(losses) + 1), moving_loss, 
                            color=color, linewidth=2, label=name)
                else:
                    ax3.plot(losses, color=color, linewidth=2, label=name)
        
        ax3.set_title('Training Loss (Moving Average)')
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance summary
        ax4 = axes[1, 1]
        algorithms = []
        avg_rewards = []
        final_rewards = []
        
        for name in colors.keys():
            if name in self.results and self.results[name]['rewards']:
                rewards = self.results[name]['rewards']
                algorithms.append(name)
                avg_rewards.append(np.mean(rewards))
                final_rewards.append(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax4.bar(x - width/2, avg_rewards, width, label='Overall Average', alpha=0.7)
        ax4.bar(x + width/2, final_rewards, width, label='Final 10 Episodes', alpha=0.7)
        
        ax4.set_title('Performance Summary')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Average Reward')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algorithms, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{self.game_name.lower()}_learning_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curves saved to: {plot_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print detailed results summary"""
        print(f"\n{'='*60}")
        print(f"{self.game_name.upper()} EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        # Sort algorithms by final performance
        performance = []
        for name, data in self.results.items():
            if data['rewards']:
                final_perf = np.mean(data['rewards'][-10:]) if len(data['rewards']) >= 10 else np.mean(data['rewards'])
                performance.append((name, final_perf, data))
        
        performance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<5} {'Algorithm':<15} {'Avg Reward':<12} {'Final 10':<12} {'Best':<12} {'Time':<8}")
        print("-" * 70)
        
        for i, (name, final_perf, data) in enumerate(performance):
            avg_reward = np.mean(data['rewards'])
            best_reward = max(data['rewards'])
            elapsed = data['elapsed']
            
            print(f"#{i+1:<4} {name:<15} {avg_reward:>8.2f}     {final_perf:>8.2f}     {best_reward:>8.2f}     {elapsed:>6.1f}s")
        
        # Learning progress analysis
        print(f"\nLearning Progress Analysis:")
        print("-" * 30)
        for name, _, data in performance:
            rewards = data['rewards']
            if len(rewards) >= 20:
                early = np.mean(rewards[:10])
                late = np.mean(rewards[-10:])
                improvement = late - early
                print(f"{name:<15}: {early:>6.2f} → {late:>6.2f} (Δ{improvement:>+6.2f})")


def main():
    parser = argparse.ArgumentParser(description='Single Game Algorithm Comparison')
    parser.add_argument('--game', choices=['ALE/Breakout-v5', 'ALE/SpaceInvaders-v5', 'ALE/Pong-v5', 'ALE/Alien-v5'], 
                       default='ALE/Breakout-v5', help='Game to test')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Episodes per algorithm (default: 100)')
    parser.add_argument('--output-dir', type=str, default='./results/single_game',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        experiment = SingleGameExperiment(
            game=args.game,
            episodes=args.episodes,
            output_dir=args.output_dir
        )
        
        experiment.run_all_algorithms()
        experiment.create_learning_curves()
        experiment.print_summary()
        
        print(f"\n{'='*60}")
        print("SINGLE GAME EXPERIMENT COMPLETED!")
        print(f"Check {experiment.output_dir} for results")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()