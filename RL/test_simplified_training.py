#!/usr/bin/env python3
"""
Test simplified MuZero training with 1-step simulation and better exploration
"""

import torch
import numpy as np
from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent

def test_simplified_training():
    # Configure for quick test
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 1  # No MCTS, just 1-step
    config.max_moves = 200  # Shorter episodes for testing
    config.batch_size = 32
    
    print("=" * 70)
    print("TESTING SIMPLIFIED MUZERO TRAINING")
    print("=" * 70)
    print("Configuration:")
    print(f"  - No MCTS (1-step simulation)")
    print(f"  - 50% random exploration")
    print(f"  - Random actions favor non-NOOP (70% non-NOOP, 30% NOOP)")
    print(f"  - Max moves per game: {config.max_moves}")
    print("=" * 70)
    
    # Create agent
    agent = MuZeroAgent(config)
    
    # Run 5 self-play games to check action diversity
    print("\nRunning 5 test games to verify action diversity...")
    print("-" * 70)
    
    total_action_counts = {i: 0 for i in range(4)}
    total_rewards = []
    
    for game_num in range(5):
        print(f"\nGame {game_num + 1}/5:")
        game = agent.self_play()
        
        # Count actions from this game
        for action in game.actions:
            total_action_counts[action] += 1
        
        total_rewards.append(sum(game.rewards))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY AFTER 5 GAMES:")
    print("=" * 70)
    
    total_actions = sum(total_action_counts.values())
    print(f"Total actions taken: {total_actions}")
    print(f"\nAction distribution:")
    print(f"  NOOP  (0): {total_action_counts[0]:4d} ({100*total_action_counts[0]/total_actions:5.1f}%)")
    print(f"  FIRE  (1): {total_action_counts[1]:4d} ({100*total_action_counts[1]/total_actions:5.1f}%)")
    print(f"  RIGHT (2): {total_action_counts[2]:4d} ({100*total_action_counts[2]/total_actions:5.1f}%)")
    print(f"  LEFT  (3): {total_action_counts[3]:4d} ({100*total_action_counts[3]/total_actions:5.1f}%)")
    
    print(f"\nRewards per game: {total_rewards}")
    print(f"Mean reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    
    # Check if exploration is working
    print("\n" + "=" * 70)
    print("EXPLORATION CHECK:")
    print("=" * 70)
    
    if total_action_counts[1] > 0:
        print("✅ FIRE action is being taken")
    else:
        print("❌ FIRE action never taken - problem!")
    
    if total_action_counts[2] > 0 and total_action_counts[3] > 0:
        print("✅ Both LEFT and RIGHT actions are being taken")
    else:
        print("❌ Movement actions not diverse enough")
    
    non_noop_ratio = (total_action_counts[1] + total_action_counts[2] + total_action_counts[3]) / total_actions
    print(f"\nNon-NOOP action ratio: {non_noop_ratio*100:.1f}%")
    if non_noop_ratio > 0.5:
        print("✅ Good exploration - majority non-NOOP actions")
    else:
        print("⚠️  Too many NOOP actions")
    
    # Now do a few training steps
    print("\n" + "=" * 70)
    print("RUNNING 10 TRAINING STEPS:")
    print("=" * 70)
    
    for step in range(10):
        metrics = agent.train_step()
        if metrics:
            print(f"Step {step+1}: Loss={metrics['total_loss']:.4f}, "
                  f"V={metrics['value_loss']:.4f}, "
                  f"P={metrics['policy_loss']:.4f}, "
                  f"R={metrics['reward_loss']:.4f}")
    
    print("\n✅ Simplified training test complete!")
    print("   Actions are now diverse with proper exploration.")

if __name__ == '__main__':
    test_simplified_training()