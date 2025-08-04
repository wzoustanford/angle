import numpy as np
import matplotlib.pyplot as plt
from typing import List

from model import DQNAgent, AgentConfig

def plot_training_results(episode_rewards: List[float], losses: List[float]):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6)
    if len(episode_rewards) >= 100:
        # Moving average
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(episode_rewards)), moving_avg, 'r', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    # Plot losses
    if losses:
        ax2.plot(losses)
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/training_results.png')
    plt.show()

def main():
    """Main training function"""
    # Create configuration
    config = AgentConfig()
    
    # Create agent
    agent = DQNAgent(config)
    
    # Train agent
    print("Starting training...")
    episode_rewards, losses = agent.train(num_episodes=100)
    
    # Plot results
    plot_training_results(episode_rewards, losses)
    
    # Test agent
    print("\nTesting trained agent...")
    test_rewards = agent.test(num_episodes=5, render=True)
    print(f"Average test reward: {np.mean(test_rewards):.2f}")

if __name__ == "__main__":
    main()