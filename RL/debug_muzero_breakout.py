#!/usr/bin/env python3
"""
Debug MuZero on Breakout
"""

import torch
import numpy as np
import gymnasium as gym
import ale_py
from config.MuZeroConfig import MuZeroConfig
from model.muzero_network import MuZeroNetwork
from model.muzero_mcts import MCTS
from model.muzero_agent import MuZeroAgent
from PIL import Image
import time

# Register Atari environments
gym.register_envs(ale_py)

def test_breakout_components():
    """Test individual components with Breakout"""
    print("Testing MuZero components with Breakout...")
    
    # Create config
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 5  # Very few simulations for debugging
    config.batch_size = 4  # Small batch
    
    # Create environment
    env = gym.make(config.env_name)
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space.n}")
    
    # Get observation
    obs, _ = env.reset()
    print(f"✓ Raw observation shape: {obs.shape}")
    print(f"  Min value: {obs.min()}, Max value: {obs.max()}")
    
    # Test preprocessing
    def preprocess_observation(observation):
        # Resize observation to configured size
        img = Image.fromarray(observation)
        img = img.resize((config.observation_shape[2], 
                        config.observation_shape[1]))
        observation = np.array(img)
        # Transpose to channels-first and normalize
        observation = observation.transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(observation)
    
    processed_obs = preprocess_observation(obs)
    print(f"✓ Preprocessed observation shape: {processed_obs.shape}")
    print(f"  Min value: {processed_obs.min():.3f}, Max value: {processed_obs.max():.3f}")
    
    # Test network
    config.action_space_size = env.action_space.n
    network = MuZeroNetwork(config).cuda()
    print(f"✓ Network created")
    
    # Test initial inference
    with torch.no_grad():
        start_time = time.time()
        output = network.initial_inference(processed_obs.unsqueeze(0).cuda())
        print(f"✓ Initial inference completed in {time.time() - start_time:.3f}s")
        print(f"  State shape: {output['state'].shape}")
        print(f"  Policy shape: {output['policy_logits'].shape}")
        print(f"  Value: {output['value'].item():.3f}")
    
    # Test MCTS
    mcts = MCTS(config)
    print(f"\n✓ Running MCTS with {config.num_simulations} simulations...")
    start_time = time.time()
    result = mcts.run(processed_obs.cuda(), network, temperature=1.0)
    mcts_time = time.time() - start_time
    print(f"✓ MCTS completed in {mcts_time:.3f}s")
    print(f"  Selected action: {result['action']}")
    print(f"  Time per simulation: {mcts_time/config.num_simulations:.3f}s")
    
    return True


def test_full_episode():
    """Test a full episode with timing"""
    print("\n" + "="*60)
    print("Testing full episode on Breakout...")
    print("="*60)
    
    config = MuZeroConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.num_simulations = 5  # Very few for speed
    config.batch_size = 4
    
    agent = MuZeroAgent(config)
    
    print("\nPlaying one episode...")
    start_time = time.time()
    
    # Track timing for different parts
    mcts_times = []
    step_times = []
    
    obs, _ = agent.env.reset()
    done = False
    total_reward = 0
    steps = 0
    max_steps = 100  # Limit steps for debugging
    
    while not done and steps < max_steps:
        # Preprocess
        obs_tensor = agent.preprocess_observation(obs).to(agent.device)
        
        # MCTS
        mcts_start = time.time()
        mcts_result = agent.mcts.run(obs_tensor, agent.network, temperature=1.0)
        mcts_times.append(time.time() - mcts_start)
        
        # Step
        step_start = time.time()
        obs, reward, terminated, truncated, _ = agent.env.step(mcts_result['action'])
        step_times.append(time.time() - step_start)
        
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        if steps % 20 == 0:
            print(f"  Step {steps}: Reward so far = {total_reward}, "
                  f"Avg MCTS time = {np.mean(mcts_times):.3f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Episode completed:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per step: {total_time/steps:.3f}s")
    print(f"  Avg MCTS time: {np.mean(mcts_times):.3f}s")
    print(f"  Avg env step time: {np.mean(step_times):.4f}s")


def main():
    try:
        # Test components
        if test_breakout_components():
            print("\n✅ Component test passed!")
        
        # Test full episode
        test_full_episode()
        
        print("\n" + "="*60)
        print("✅ Breakout debugging complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()