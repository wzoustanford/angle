#!/usr/bin/env python3
"""
Debug MuZero actions in Breakout
"""

import torch
import numpy as np
import gymnasium as gym
import ale_py
from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent

gym.register_envs(ale_py)

def debug_actions():
    # Load trained agent
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 5
    
    agent = MuZeroAgent(config)
    agent.load_checkpoint('./results/muzero_checkpoints/muzero_final_Breakout_20250815_034713.pth')
    
    # Create environment
    env = gym.make('ALE/Breakout-v5')
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT")
    
    # Test random agent first
    print("\n--- Random Agent (baseline) ---")
    obs, _ = env.reset()
    total_reward = 0
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Random agent reward after 200 steps: {total_reward}")
    
    # Test MuZero agent
    print("\n--- MuZero Agent ---")
    obs, _ = env.reset()
    total_reward = 0
    action_counts = {i: 0 for i in range(env.action_space.n)}
    
    print("\nFirst 50 actions taken:")
    for i in range(200):
        # Preprocess
        obs_tensor = agent.preprocess_observation(obs).to(agent.device)
        
        # Get action from MCTS
        with torch.no_grad():
            mcts_result = agent.mcts.run(
                obs_tensor,
                agent.network,
                temperature=0,
                add_exploration_noise=False
            )
        
        action = mcts_result['action']
        action_counts[action] += 1
        
        if i < 50:
            print(f"Step {i}: Action={action} ({'NOOP' if action==0 else 'FIRE' if action==1 else 'RIGHT' if action==2 else 'LEFT'}), Value={mcts_result['value']:.3f}")
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if reward > 0:
            print(f"  >>> Got reward: {reward} at step {i}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    
    print(f"\nMuZero agent reward: {total_reward}")
    print(f"Action distribution: {action_counts}")
    print(f"Most common action: {max(action_counts, key=action_counts.get)}")
    
    # Check if agent is always taking same action
    unique_actions = len([v for v in action_counts.values() if v > 0])
    if unique_actions == 1:
        print("\n⚠️ WARNING: Agent is only taking ONE action repeatedly!")
    elif unique_actions == 2:
        print("\n⚠️ WARNING: Agent is only using 2 actions!")
    
    # Test the network predictions directly
    print("\n--- Network Predictions ---")
    obs, _ = env.reset()
    obs_tensor = agent.preprocess_observation(obs).to(agent.device)
    
    with torch.no_grad():
        output = agent.network.initial_inference(obs_tensor.unsqueeze(0))
        policy = torch.softmax(output['policy_logits'], dim=-1).cpu().numpy()[0]
        
    print(f"Policy distribution: {policy}")
    print(f"Highest probability action: {np.argmax(policy)}")
    print(f"Value estimate: {output['value'].item():.3f}")
    
    env.close()

if __name__ == '__main__':
    debug_actions()