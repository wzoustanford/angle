import sys
import os
sys.path.insert(0, '/home/ubuntu/code/angle/RL')

from config.DistributedAgentConfig import DistributedAgentConfig
from model.distributed_dqn_agent import DistributedDQNAgent

# Quick test
config = DistributedAgentConfig()
config.env_name = 'ALE/Alien-v5'
config.num_workers = 2
config.memory_size = 5000
config.use_prioritized_replay = True

print("Testing Distributed DQN with 2 episodes...")
agent = DistributedDQNAgent(config, num_workers=2)
results = agent.train_distributed(total_episodes=2)

print(f"✓ Test completed successfully\!")
print(f"  Total episodes: {results['env_stats']['total_episodes']}")
print(f"  Average reward: {results['env_stats']['overall_avg_reward']:.2f}")
