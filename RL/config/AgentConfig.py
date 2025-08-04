from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for DQN agent"""
    # Environment
    env_name: str = 'ALE/SpaceInvaders-v5'
    frame_stack: int = 4
    
    # Training
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    learning_rate: float = 1e-4
    memory_size: int = 10000
    target_update_freq: int = 1000  # Update target network every N steps
    policy_update_interval: int = 4  # Update policy network every N steps
    min_replay_size: int = 1000
    
    # Saving
    save_interval: int = 5000
    checkpoint_dir: str = './results/checkpoints'