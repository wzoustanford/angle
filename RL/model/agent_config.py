class AgentConfig:
    """Configuration for DQN agent"""
    def __init__(self):
        # Environment
        self.env_name = 'ALE/SpaceInvaders-v5'
        self.frame_stack = 4
        
        # Training
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.memory_size = 10000
        self.target_update_freq = 1000  # Update target network every N steps
        self.policy_update_interval = 4  # Update policy network every N steps
        self.min_replay_size = 1000
        
        # Saving
        self.save_interval = 5000
        self.checkpoint_dir = './results/checkpoints'