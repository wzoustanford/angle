from .AgentConfig import AgentConfig


class NGUConfig(AgentConfig):
    """
    Configuration for Never Give Up (NGU) Agent
    
    Extends standard AgentConfig with NGU-specific parameters for:
    - Episodic memory configuration  
    - Random Network Distillation settings
    - Dual reward system parameters
    - LSTM/sequence learning settings
    """
    
    def __init__(self):
        super().__init__()
        
        # Frame stacking
        self.frame_stack = 4
        
        # Use R2D2/sequence learning by default for NGU
        self.use_r2d2 = True
        self.sequence_length = 80
        self.burn_in_length = 40
        self.lstm_size = 512
        self.num_lstm_layers = 1
        
        # NGU-specific intrinsic reward parameters
        self.embedding_dim = 128
        self.rnd_feature_dim = 512
        self.episodic_memory_size = 50000
        self.k_neighbors = 10
        
        # Dual reward system
        self.gamma_extrinsic = 0.999  # Discount for extrinsic rewards
        self.gamma_intrinsic = 0.99   # Discount for intrinsic rewards
        self.intrinsic_reward_scale = 1.0
        self.extrinsic_reward_scale = 1.0
        
        # RND training parameters
        self.rnd_update_frequency = 4  # Update RND every N steps
        self.rnd_learning_rate = 1e-4
        
        # Episodic memory parameters
        self.kernel_epsilon = 0.001    # For similarity computation
        self.cluster_distance = 0.008  # Distance threshold for clustering
        self.c_constant = 0.001        # Constant for pseudo-count calculation
        
        # Enhanced exploration (higher epsilon for more exploration)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        
        # Larger replay buffer for better sample diversity
        self.memory_size = 100000
        self.min_replay_size = 10000
        self.batch_size = 32
        
        # Learning parameters
        self.learning_rate = 1e-4
        self.max_grad_norm = 5.0
        
        # Episode parameters
        self.max_episode_steps = 10000
        
        # Training frequency
        self.policy_update_interval = 4
        self.target_update_freq = 2000
        
        # Save/load parameters
        self.save_interval = 10000
        
        # Environment-specific optimizations
        self.optimize_for_env = True


class Agent57Config(NGUConfig):
    """
    Configuration for Agent57 (multi-policy NGU)
    
    Extends NGUConfig with multi-policy learning parameters
    """
    
    def __init__(self):
        super().__init__()
        
        # Agent57-specific parameters
        self.num_policies = 32
        self.policy_schedule = 'round_robin'  # 'round_robin', 'random', 'meta_learning'
        
        # Meta-learning parameters
        self.meta_learning_rate = 1e-4
        self.policy_update_frequency = 100  # Episodes between policy updates
        
        # Per-policy memory (smaller since we have multiple)
        self.episodic_memory_size = 20000
        
        # Enhanced diversity through more policies
        self.min_replay_size = 20000  # Larger minimum for better policy diversity
        
        # Longer sequences for better temporal modeling
        self.sequence_length = 100
        self.burn_in_length = 50


def create_ngu_config_for_game(env_name: str, 
                              episodes: int = 50,
                              use_agent57: bool = False) -> NGUConfig:
    """
    Create optimized NGU config for specific games
    
    Args:
        env_name: Environment name (e.g., 'ALE/Alien-v5')
        episodes: Target number of episodes
        use_agent57: Whether to use Agent57 (multi-policy)
        
    Returns:
        Configured NGUConfig or Agent57Config
    """
    if use_agent57:
        config = Agent57Config()
        config.num_policies = min(32, max(8, episodes // 4))  # Scale policies with episodes
    else:
        config = NGUConfig()
    
    config.env_name = env_name
    
    # Game-specific optimizations
    if 'IceHockey' in env_name:
        # Ice Hockey has longer episodes - optimize for memory
        config.episodic_memory_size = max(10000, config.episodic_memory_size // 2)
        config.memory_size = max(20000, config.memory_size // 2)
        config.batch_size = max(16, config.batch_size // 2)
        config.max_episode_steps = 5000  # Limit episode length
        config.sequence_length = 60      # Shorter sequences
        config.burn_in_length = 30
        
        # More conservative exploration for Ice Hockey
        config.epsilon_start = 0.5
        config.epsilon_end = 0.05
        config.intrinsic_reward_scale = 0.5
        
        if use_agent57:
            config.num_policies = max(8, config.num_policies // 2)
            
    elif 'Alien' in env_name:
        # Alien has shorter episodes - can use full capacity
        config.max_episode_steps = 3000
        config.intrinsic_reward_scale = 1.0
        
        # Alien benefits from more exploration
        config.epsilon_start = 1.0
        config.epsilon_end = 0.1
        
    # Scale parameters based on episode count
    if episodes <= 10:
        # Quick testing with minimal memory
        config.memory_size = 5000
        config.min_replay_size = 500
        config.episodic_memory_size = 2000
        config.target_update_freq = 500
        config.batch_size = 8
        config.sequence_length = 40
        config.burn_in_length = 20
        
    elif episodes <= 50:
        # Standard experiments
        config.memory_size = max(20000, config.memory_size // 2)
        config.min_replay_size = max(5000, config.min_replay_size // 2)
        config.target_update_freq = 1000
        
    # High-level experiment parameters
    config.total_episodes = episodes
    
    return config


def test_ngu_config():
    """Test NGU configuration creation"""
    print("Testing NGU Configuration...")
    
    # Test standard NGU config
    ngu_config = NGUConfig()
    print(f"NGU Config - Memory size: {ngu_config.memory_size}")
    print(f"NGU Config - Embedding dim: {ngu_config.embedding_dim}")
    print(f"NGU Config - Sequence length: {ngu_config.sequence_length}")
    
    # Test Agent57 config  
    agent57_config = Agent57Config()
    print(f"Agent57 Config - Num policies: {agent57_config.num_policies}")
    print(f"Agent57 Config - Policy schedule: {agent57_config.policy_schedule}")
    
    # Test game-specific configs
    alien_config = create_ngu_config_for_game('ALE/Alien-v5', episodes=20)
    print(f"Alien Config - Max episode steps: {alien_config.max_episode_steps}")
    print(f"Alien Config - Intrinsic reward scale: {alien_config.intrinsic_reward_scale}")
    
    ice_hockey_config = create_ngu_config_for_game('ALE/IceHockey-v5', episodes=20)
    print(f"Ice Hockey Config - Max episode steps: {ice_hockey_config.max_episode_steps}")
    print(f"Ice Hockey Config - Memory size: {ice_hockey_config.memory_size}")
    
    # Test Agent57 for games
    agent57_alien_config = create_ngu_config_for_game('ALE/Alien-v5', episodes=50, use_agent57=True)
    print(f"Agent57 Alien Config - Num policies: {agent57_alien_config.num_policies}")
    
    print("✓ NGU Configuration test passed!")


if __name__ == "__main__":
    test_ngu_config()