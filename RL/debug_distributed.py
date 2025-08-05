#!/usr/bin/env python3
"""
Debug script for distributed agent issue
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def debug_distributed_collection():
    """Debug what's happening with distributed collection"""
    print("Debugging Distributed Agent Collection...")
    
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Pong-v5'
    config.num_workers = 2
    config.memory_size = 1000
    config.min_replay_size = 100
    config.save_interval = 50000
    
    agent = DistributedDQNAgent(config, num_workers=2)
    
    print("Starting collection...")
    start_time = time.time()
    
    # Start collection manually to debug
    agent.env_manager.start_collection(episodes_per_worker=3)  # Fixed number
    
    # Monitor for a while
    for i in range(30):  # Check for 30 seconds
        time.sleep(1)
        stats = agent.env_manager.get_statistics()
        print(f"Second {i+1}: Total episodes: {stats.get('total_episodes', 0)}, "
              f"Buffer size: {len(agent.replay_buffer)}")
        
        if stats.get('total_episodes', 0) >= 6:  # 3 per worker
            print("Target episodes reached!")
            break
    
    # Stop collection
    agent.env_manager.stop_collection()
    
    final_stats = agent.env_manager.get_statistics()
    print(f"\nFinal stats: {final_stats}")
    
    elapsed = time.time() - start_time
    print(f"Debug completed in {elapsed:.1f}s")


if __name__ == '__main__':
    debug_distributed_collection()