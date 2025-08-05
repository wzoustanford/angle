from dataclasses import dataclass


@dataclass
class DistributedAgentConfig:
    """Configuration for Distributed DQN agent"""
    # Environment
    env_name: str = 'ALE/SpaceInvaders-v5'
    frame_stack: int = 4
    
    # Distributed settings
    num_workers: int = 4
    collection_mode: str = 'continuous'  # 'continuous' or 'batch'
    episodes_per_batch: int = 20  # For batch mode
    prioritize_recent_experiences: bool = True
    
    # Training
    batch_size: int = 64  # Larger batch size for distributed training
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  # Lower minimum for distributed
    epsilon_decay: float = 0.9995  # Slower decay with more data
    learning_rate: float = 1e-4
    memory_size: int = 50000  # Larger buffer for distributed collection
    target_update_freq: int = 2000  # Less frequent updates
    policy_update_interval: int = 8  # More updates per step
    min_replay_size: int = 5000  # Larger minimum size
    
    # Saving and monitoring
    save_interval: int = 10000
    stats_interval: int = 30  # Print stats every 30 seconds
    checkpoint_dir: str = './results/distributed_checkpoints'
    
    # Performance tuning
    max_episode_length: int = 10000
    worker_epsilon_schedule: bool = True  # Use per-worker epsilon schedules