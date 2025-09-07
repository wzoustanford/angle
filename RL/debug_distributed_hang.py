#!/usr/bin/env python3
"""Debug script to identify where distributed training hangs"""

import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(__file__))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent

def timeout_handler(signum, frame):
    print("\n⏰ TIMEOUT: Process took too long")
    raise TimeoutError("Operation timed out")

def test_distributed():
    """Test basic distributed training"""
    
    print("="*60)
    print("DEBUGGING DISTRIBUTED TRAINING HANG")
    print("="*60)
    
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.num_workers = 2
    config.memory_size = 1000
    config.batch_size = 16
    config.min_replay_size = 100
    
    print(f"Config: {config.num_workers} workers, buffer={config.memory_size}")
    
    try:
        # Set timeout for initialization
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        print("\n1. Initializing agent...")
        agent = DistributedDQNAgent(config, num_workers=2)
        print("   ✓ Agent initialized")
        
        # Reset timeout for first training
        signal.alarm(60)  # 60 second timeout
        
        print("\n2. Running first training batch (2 episodes)...")
        start = time.time()
        result1 = agent.train_distributed(total_episodes=2)
        print(f"   ✓ Completed in {time.time()-start:.1f}s")
        print(f"   Episodes: {result1.get('env_stats', {}).get('total_episodes', 0)}")
        
        # Reset timeout for second training
        signal.alarm(60)
        
        print("\n3. Running second training batch (2 episodes)...")
        start = time.time()
        result2 = agent.train_distributed(total_episodes=2)
        print(f"   ✓ Completed in {time.time()-start:.1f}s")
        print(f"   Episodes: {result2.get('env_stats', {}).get('total_episodes', 0)}")
        
        # Disable timeout
        signal.alarm(0)
        
        print("\n✅ All tests passed! Distributed training works")
        return True
        
    except TimeoutError as e:
        print(f"\n❌ {e}")
        print("Training hung at some point")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)  # Disable any remaining timeout

if __name__ == '__main__':
    test_distributed()