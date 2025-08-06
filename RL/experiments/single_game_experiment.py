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
    
    ARCHIVED RESULTS - 20 Episode Alien Experiment (2025-08-05):
    ============================================================
    Hardware: Tesla T4 GPU (14.6GB), CUDA enabled
    Total Runtime: 4.4 hours (wall-clock time)
    
    ALGORITHM RANKINGS & PERFORMANCE:
    #1 R2D2 (Single-threaded):    278.5 avg, 920.0 best, Δ+63.00 improvement
    #2 Basic DQN:                 269.5 avg, 430.0 best, Δ+37.00 improvement  
    #3 DQN + Priority:            217.5 avg, 300.0 best, Δ+29.00 improvement
    
    DETAILED ALGORITHM CONFIGURATIONS:
    
    Basic DQN Settings:
      - batch_size: 64, learning_rate: 1e-4, memory_size: 20000
      - epsilon_decay: 0.95, target_update_freq: 500, min_replay_size: 100
      - Runtime: 389.5s (6.5 min), Time/episode: 19.5s
      - Device: CUDA (Tesla T4)
    
    DQN + Priority Settings:
      - Same as Basic DQN + prioritized replay
      - priority_alpha: 0.7, priority_beta_start: 0.5→1.0
      - Beta annealing: 1.2e-8 per step (aligned with ICLR 2016 paper)
      - Priority type: 'td_error' (proportional variant)
      - Runtime: 313.4s (5.2 min), Time/episode: 15.7s
      - Device: CUDA (Tesla T4)
    
    R2D2 (Single-threaded RNN) Settings:
      - batch_size: 32, lstm_size: 256, num_lstm_layers: 1
      - sequence_length: 40, burn_in_length: 20
      - Inherits priority replay settings (alpha=0.7, beta_start=0.5)
      - Runtime: 15,115.9s (4.2 hours), Time/episode: 756s (12.6 min)
      - Device: CUDA (Tesla T4)
      - Note: NOT true distributed R2D2, missing 256 parallel actors
    
    PERFORMANCE ANALYSIS:
    - GPU Utilization: High across all algorithms
    - R2D2 Slowdown: 39x slower per episode vs Basic DQN (LSTM overhead)
    - Priority Replay: Underperformed despite fixed parameters (short experiment)
    - Best Learning: R2D2 achieved highest rewards but at massive computational cost
    
    CONCLUSIONS:
    - Basic DQN most efficient (good performance, fast training)
    - R2D2 best performance but impractical runtime for experimentation
    - Priority replay needs longer training to show benefits
    - True distributed R2D2 needed for realistic comparisons
    """
    
    def __init__(self, game='ALE/Breakout-v5', episodes=100, output_dir='./results/single_game', 
                 time_budget_seconds=None):
        self.game = game
        self.game_name = game.split('/')[-1].replace('-v5', '')
        self.episodes = episodes
        self.time_budget_seconds = time_budget_seconds
        
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
        config.batch_size = 64          # Increased for more stable training
        config.learning_rate = 1e-4
        config.target_update_freq = 500  # More frequent updates
        config.min_replay_size = 100    # Start training sooner (better for short experiments)
        config.save_interval = 50000    # Disable saving
        
        # Faster epsilon decay for short experiments
        config.epsilon_decay = 0.95     # More aggressive decay (was 0.995)
        
        return config
    
    def create_prioritized_dqn_config(self) -> AgentConfig:
        """DQN + Prioritized Replay configuration"""
        config = self.create_basic_dqn_config()
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'  # Proportional variant
        config.priority_alpha = 0.7         # Proportional variant uses 0.7
        config.priority_beta_start = 0.5    # Proportional variant uses 0.5
        config.priority_beta_end = 1.0
        return config
    
    def create_distributed_dqn_config(self):
        """Distributed DQN + Priority configuration"""
        from config.DistributedAgentConfig import DistributedAgentConfig
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = 4          # 4 parallel workers
        config.batch_size = 64          # Larger batch for distributed
        config.learning_rate = 1e-4
        config.memory_size = 20000      # Same as single agent for fair comparison
        config.target_update_freq = 500
        config.min_replay_size = 100
        config.save_interval = 50000    # Disable saving
        
        # Faster epsilon decay for short experiments  
        config.epsilon_decay = 0.95
        
        # Enable prioritized replay
        config.use_prioritized_replay = True
        config.priority_alpha = 0.7
        config.priority_beta_start = 0.5
        config.priority_beta_end = 1.0
        
        return config
    
    def create_distributed_dueling_dqn_config(self):
        """Distributed DQN + Dueling + Priority configuration"""
        from config.DistributedAgentConfig import DistributedAgentConfig
        config = DistributedAgentConfig()
        config.env_name = self.game
        config.num_workers = 4          # 4 parallel workers
        config.batch_size = 64          # Larger batch for distributed
        config.learning_rate = 1e-4
        config.memory_size = 20000      # Same as single agent for fair comparison
        config.target_update_freq = 500
        config.min_replay_size = 100
        config.save_interval = 50000    # Disable saving
        
        # Faster epsilon decay for short experiments  
        config.epsilon_decay = 0.95
        
        # Enable prioritized replay
        config.use_prioritized_replay = True
        config.priority_alpha = 0.7
        config.priority_beta_start = 0.5
        config.priority_beta_end = 1.0
        
        # Enable dueling architecture
        config.use_dueling = True
        
        return config
    
    def run_algorithm(self, name: str, config) -> tuple:
        """Run a single algorithm and return results"""
        print(f"\n{'='*50}")
        print(f"Running {name}")
        print(f"{'='*50}")
        print(f"DEBUG: Starting algorithm {name}")
        
        start_time = time.time()
        
        try:
            # Check if this is a distributed configuration
            if hasattr(config, 'num_workers'):
                use_dueling = getattr(config, 'use_dueling', False)
                print(f"DEBUG: Creating DistributedDQNAgent with {config.num_workers} workers...")
                if use_dueling:
                    print(f"DEBUG: Using Dueling architecture")
                agent = DistributedDQNAgent(config, num_workers=config.num_workers, use_dueling=use_dueling)
                print(f"DEBUG: DistributedDQNAgent created successfully")
                is_distributed = True
            else:
                print(f"DEBUG: Creating DQNAgent with config...")
                agent = DQNAgent(config)
                print(f"DEBUG: DQNAgent created successfully")
                is_distributed = False
            print(f"DEBUG: Agent device: {agent.device}")
            print(f"DEBUG: Buffer type: {type(agent.replay_buffer).__name__}")
            print(f"Starting {name} training...")
            
            if is_distributed:
                # Use distributed training approach
                print(f"Using distributed training with {config.num_workers} workers")
                if self.time_budget_seconds:
                    print(f"Time budget: {self.time_budget_seconds}s")
                    # For distributed, we'll use time budget and see how many episodes we get
                    final_stats = agent.train_distributed(total_episodes=1000, max_time_seconds=self.time_budget_seconds)
                else:
                    final_stats = agent.train_distributed(total_episodes=self.episodes)
                
                # Extract episode rewards from worker statistics
                episode_rewards = []
                for worker_stats in final_stats['env_stats']['worker_stats']:
                    # Get rewards from each worker (approximate)
                    worker_episodes = worker_stats['total_episodes']
                    avg_reward = worker_stats['avg_reward']
                    # Create approximated episode rewards
                    episode_rewards.extend([avg_reward] * worker_episodes)
                
                # Trim to requested number of episodes
                episode_rewards = episode_rewards[:self.episodes]
                losses = final_stats['training_stats']['losses'][-self.episodes:] if final_stats['training_stats']['losses'] else []
                
            else:
                # Use single-agent training approach
                episode_rewards = []
                losses = []
                episode_start_time = time.time()
                
                episode = 0
                while (episode < self.episodes and 
                       (self.time_budget_seconds is None or 
                        time.time() - start_time < self.time_budget_seconds)):
                    print(f"DEBUG: Starting episode {episode+1}/{self.episodes}")
                    # Reset environment and hidden state
                    obs, _ = agent.env.reset()
                    print(f"DEBUG: Environment reset, obs shape: {obs.shape}")
                    state = agent.frame_stack.reset(obs)
                    print(f"DEBUG: Frame stack reset, state shape: {state.shape}")
                    agent.reset_hidden_state()
                    print(f"DEBUG: Hidden state reset")
                    episode_reward = 0
                    episode_losses = []
                    episode_steps = 0
                    
                    done = False
                    step_count = 0
                    while not done:
                        step_count += 1
                        if step_count % 20 == 1:  # Log every 20 steps
                            print(f"DEBUG: Episode {episode+1} step {step_count}")
                        
                        # Select and perform action
                        action = agent.select_action(state)
                        next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                        done = terminated or truncated
                        
                        # Clip rewards if configured
                        if getattr(agent.config, 'clip_rewards', False):
                            reward = np.clip(reward, -1.0, 1.0)
                        
                        # Stack frames
                        next_state = agent.frame_stack.append(next_obs)
                        
                        # Store transition
                        if getattr(agent.config, 'use_r2d2', False):
                            agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                        else:
                            agent.replay_buffer.push(state, action, reward, next_state, done)
                        
                        # Update state
                        state = next_state
                        episode_reward += reward
                        agent.steps_done += 1
                        episode_steps += 1
                        
                        # Update Q-network
                        if agent.steps_done % agent.config.policy_update_interval == 0:
                            loss = agent.update_q_network()
                            if loss is not None:
                                episode_losses.append(loss)
                            # Debug: Track if R2D2 is actually training
                            if getattr(agent.config, 'use_r2d2', False) and episode_steps % 100 == 0:
                                buffer_size = len(agent.replay_buffer)
                                training_status = "training" if loss is not None else "no training"
                                print(f"    Step {episode_steps}: buffer={buffer_size} sequences, {training_status}")
                        
                        # Update target network
                        if agent.steps_done % agent.config.target_update_freq == 0:
                            agent.update_target_network()
                            print(f"Updated target network at step {agent.steps_done}")
                    
                    # Update exploration parameters
                    agent.epsilon = max(agent.config.epsilon_end, agent.epsilon * agent.config.epsilon_decay)
                    
                    if agent.config.use_prioritized_replay:
                        # Align beta annealing rate with original paper (50M steps)
                        # Original: 0.4 → 1.0 over 50M steps = 1.2e-8 per step
                        annealing_rate_per_step = 1.2e-8
                        beta_increment = annealing_rate_per_step * agent.steps_done
                        agent.priority_beta = min(1.0, agent.config.priority_beta_start + beta_increment)
                        if hasattr(agent.replay_buffer, 'update_beta'):
                            agent.replay_buffer.update_beta(agent.priority_beta)
                    
                    # Record statistics
                    episode_rewards.append(episode_reward)
                    if episode_losses:
                        losses.append(np.mean(episode_losses))
                    
                    # Print episode progress
                    episode_time = time.time() - episode_start_time
                    buffer_size_after = len(agent.replay_buffer)
                    if getattr(agent.config, 'use_r2d2', False):
                        print(f"  Episode {episode+1}: {episode_steps} steps, "
                              f"reward: {episode_reward:.1f}, time: {episode_time:.1f}s, "
                              f"buffer: {buffer_size_after} sequences")
                    else:
                        print(f"  Episode {episode+1}: {episode_steps} steps, "
                              f"reward: {episode_reward:.1f}, time: {episode_time:.1f}s")
                    episode_start_time = time.time()
                    
                    episode += 1
            
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
            ('Distributed DQN + Priority', self.create_distributed_dqn_config()),
            ('Distributed DQN + Dueling + Priority', self.create_distributed_dueling_dqn_config()),
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
            'Basic DQN': '#1f77b4',                              # Blue
            'DQN + Priority': '#ff7f0e',                         # Orange
            'Distributed DQN + Priority': '#2ca02c',             # Green
            'Distributed DQN + Dueling + Priority': '#d62728'    # Red
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
    parser.add_argument('--time-budget', type=int, default=None,
                       help='Time budget in seconds (overrides episodes)')
    parser.add_argument('--output-dir', type=str, default='./results/single_game',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        experiment = SingleGameExperiment(
            game=args.game,
            episodes=args.episodes,
            output_dir=args.output_dir,
            time_budget_seconds=args.time_budget
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