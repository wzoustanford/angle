#!/usr/bin/env python3
"""
Debug self-play to find where it's getting stuck
"""

import torch
import numpy as np
import time
from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent
from model.muzero_buffer import GameHistory

def debug_self_play():
    print("Debugging self-play...")
    
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 3  # Very few
    config.max_moves = 100  # Very short episode
    
    agent = MuZeroAgent(config)
    
    print(f"Config:")
    print(f"  Max moves: {config.max_moves}")
    print(f"  Simulations: {config.num_simulations}")
    print(f"  Temperature function: {config.visit_softmax_temperature_fn}")
    
    # Manually do self-play with timing
    game = GameHistory()
    observation, _ = agent.env.reset()
    done = False
    
    # Test temperature function
    print(f"\nTesting temperature function...")
    try:
        temp = config.visit_softmax_temperature_fn(0)
        print(f"  Temperature at step 0: {temp}")
    except Exception as e:
        print(f"  Error with temperature function: {e}")
        # Fix it
        agent.visit_softmax_temperature_fn = lambda x: 1.0
        print(f"  Using fixed temperature: 1.0")
    
    print(f"\nStarting self-play loop...")
    step_count = 0
    
    while not done and step_count < config.max_moves:
        step_start = time.time()
        
        # Preprocess observation
        obs_tensor = agent.preprocess_observation(observation).to(agent.device)
        
        # Run MCTS
        print(f"  Step {step_count}: Running MCTS...", end="")
        mcts_start = time.time()
        temperature = 1.0  # Fixed temperature for debugging
        mcts_result = agent.mcts.run(
            obs_tensor,
            agent.network,
            temperature=temperature,
            add_exploration_noise=True
        )
        mcts_time = time.time() - mcts_start
        print(f" done in {mcts_time:.3f}s")
        
        # Store in game history
        game.store(
            observation=obs_tensor.cpu().numpy(),
            action=mcts_result['action'],
            reward=0,
            policy=mcts_result['action_probs'],
            value=mcts_result['value']
        )
        
        # Take action in environment
        observation, reward, terminated, truncated, _ = agent.env.step(mcts_result['action'])
        done = terminated or truncated
        
        # Update reward
        if len(game.rewards) > 0:
            game.rewards[-1] = reward
        
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"    Progress: {step_count} steps, total reward: {sum(game.rewards)}")
    
    print(f"\nSelf-play completed:")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {sum(game.rewards)}")
    print(f"  Done: {done}")
    
    # Test saving to buffer
    print(f"\nSaving to replay buffer...")
    agent.replay_buffer.save_game(game)
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    # Test sampling
    if agent.replay_buffer.is_ready():
        print(f"\nTesting batch sampling...")
        batch = agent.replay_buffer.sample_batch()
        print(f"  Batch observations shape: {batch['observations'].shape}")
        print(f"  Batch actions shape: {batch['actions'].shape}")
        print(f"  Batch target values shape: {batch['target_values'].shape}")
    
    print("\n✅ Self-play debugging complete!")

if __name__ == '__main__':
    debug_self_play()