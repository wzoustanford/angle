#!/usr/bin/env python3
"""
Train MuZero on Atari games
"""

import argparse
import torch
import numpy as np
import random
import os
from datetime import datetime

from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train MuZero on Atari games')
    parser.add_argument('--game', type=str, default='SpaceInvaders',
                       help='Atari game to train on (e.g., SpaceInvaders, Breakout, Alien)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of training iterations (overrides episodes if set)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (default: use iterations)')
    parser.add_argument('--simulations', type=int, default=50,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda/cuda:0, etc.)')
    parser.add_argument('--checkpoint-dir', type=str, default='./results/muzero_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run evaluation, no training')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--train-steps-per-episode', type=int, default=50,
                       help='Number of training steps to perform per episode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.iterations is None and args.episodes is None:
        args.episodes = 100  # Default to 100 episodes
        print(f"No iterations or episodes specified, defaulting to {args.episodes} episodes")
    elif args.iterations is not None and args.episodes is not None:
        print(f"Both iterations and episodes specified, using iterations: {args.iterations}")
        args.episodes = None
    
    # Set random seed
    set_seed(args.seed)
    
    # Create config
    config = MuZeroConfig()
    
    # Update config from command line arguments
    config.env_name = f'ALE/{args.game}-v5'
    config.num_simulations = args.simulations
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    config.checkpoint_dir = args.checkpoint_dir
    
    # Create agent
    print(f"Initializing MuZero agent for {args.game}...")
    agent = MuZeroAgent(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        agent.load_checkpoint(args.load_checkpoint)
        print(f"Resumed from checkpoint: {args.load_checkpoint}")
    
    # Test only mode
    if args.test_only:
        print("Running evaluation...")
        eval_metrics = agent.evaluate(num_episodes=10, render=args.render)
        print("\nEvaluation Results:")
        print(f"  Mean reward: {eval_metrics['mean_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
        print(f"  Max reward: {eval_metrics['max_reward']:.1f}")
        print(f"  Min reward: {eval_metrics['min_reward']:.1f}")
        print(f"  Mean episode length: {eval_metrics['mean_length']:.1f}")
        return
    
    # Training
    print(f"\nStarting MuZero training on {args.game}")
    print(f"Configuration:")
    print(f"  Environment: {config.env_name}")
    print(f"  MCTS simulations: {config.num_simulations}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {agent.device}")
    print(f"  Checkpoint directory: {config.checkpoint_dir}")
    if args.episodes:
        print(f"  Training mode: Episode-based ({args.episodes} episodes)")
        print(f"  Train steps per episode: {args.train_steps_per_episode}")
    else:
        print(f"  Training mode: Iteration-based ({args.iterations} iterations)")
    print("-" * 50)
    
    try:
        if args.episodes:
            # Episode-based training
            agent.train_episodes(
                num_episodes=args.episodes,
                train_steps_per_episode=args.train_steps_per_episode
            )
        else:
            # Iteration-based training
            agent.train(num_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_metrics = agent.evaluate(num_episodes=10, render=args.render)
    print("\nFinal Results:")
    print(f"  Mean reward: {eval_metrics['mean_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
    print(f"  Max reward: {eval_metrics['max_reward']:.1f}")
    print(f"  Games played: {agent.games_played}")
    print(f"  Training steps: {agent.training_steps}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"muzero_final_{args.game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    agent.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}")


if __name__ == '__main__':
    main()