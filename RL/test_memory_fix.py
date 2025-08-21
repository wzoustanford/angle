#!/usr/bin/env python3
"""Test memory leak fixes in R2D2+Agent57"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import Agent57Config
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid
from model.r2d2_agent57_fixed import R2D2Agent57Fixed
import psutil
import gc
import torch

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def test_original():
    """Test original implementation"""
    print("Testing ORIGINAL R2D2+Agent57")
    print("="*50)
    
    config = Agent57Config()
    config.env_name = 'ALE/Alien-v5'
    config.num_policies = 4
    config.memory_size = 2000
    config.episodic_memory_size = 500
    config.sequence_length = 20
    config.burn_in_length = 10
    config.max_episode_steps = 500
    
    initial_mem = get_mem_mb()
    print(f"Initial memory: {initial_mem:.1f}MB\n")
    
    try:
        agent = R2D2Agent57Hybrid(config)
        after_init = get_mem_mb()
        print(f"After init: {after_init:.1f}MB (+{after_init-initial_mem:.1f}MB)\n")
        
        for ep in range(3):
            stats = agent.train_episode()
            current_mem = get_mem_mb()
            print(f"Episode {ep+1}: Reward={stats['episode_reward']:.0f}, "
                  f"Memory={current_mem:.0f}MB (+{current_mem-initial_mem:.0f})")
        
        final_mem = get_mem_mb()
        print(f"\nMemory per episode: {(final_mem-after_init)/3:.1f}MB")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_fixed():
    """Test fixed implementation"""
    print("\nTesting FIXED R2D2+Agent57")
    print("="*50)
    
    config = Agent57Config()
    config.env_name = 'ALE/Alien-v5'
    config.num_policies = 4
    config.memory_size = 2000
    config.episodic_memory_size = 500
    config.sequence_length = 20
    config.burn_in_length = 10
    config.max_episode_steps = 500
    
    initial_mem = get_mem_mb()
    print(f"Initial memory: {initial_mem:.1f}MB\n")
    
    try:
        agent = R2D2Agent57Fixed(config)
        after_init = get_mem_mb()
        print(f"After init: {after_init:.1f}MB (+{after_init-initial_mem:.1f}MB)\n")
        
        for ep in range(3):
            stats = agent.train_episode()
            current_mem = get_mem_mb()
            print(f"Episode {ep+1}: Reward={stats['episode_reward']:.0f}, "
                  f"Memory={current_mem:.0f}MB (+{current_mem-initial_mem:.0f})")
        
        final_mem = get_mem_mb()
        print(f"\nMemory per episode: {(final_mem-after_init)/3:.1f}MB")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Force garbage collection between tests
    test_original()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    test_fixed()
    
    print("\n" + "="*50)
    print("Memory test complete!")