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
    
    # Prioritized Replay Buffer (ICLR 2016)
    # Set use_prioritized_replay=True to enable prioritized experience replay
    # This samples experiences based on their "importance" rather than uniformly
    use_prioritized_replay: bool = False
    
    # Priority calculation method - determines what makes an experience "important"
    priority_type: str = 'td_error'  # Options:
                                     # 'td_error': High TD-error = high priority (recommended)
                                     # 'reward': High absolute reward = high priority
                                     # 'random': Random priorities (for comparison)
    
    # Prioritization strength: how much to favor high-priority experiences
    priority_alpha: float = 0.6     # Range: 0.0 to 1.0
                                     # 0.0 = uniform sampling (like standard replay)
                                     # 1.0 = full prioritization (only sample highest priorities)
                                     # 0.6 = recommended balance
    
    # Importance sampling bias correction (anneals during training)
    priority_beta_start: float = 0.4  # Start value (typically 0.4)
    priority_beta_end: float = 1.0    # End value (should be 1.0 for full correction)
                                       # Beta increases linearly from start to end during training
    
    # Small constant added to all priorities to prevent zero values
    priority_epsilon: float = 1e-6    # Very small positive number
    
    # Saving
    save_interval: int = 5000
    checkpoint_dir: str = './results/checkpoints'