# config.py
import torch

class Config:
    """Base configuration class for all RL algorithms"""
    def __init__(self):
        # Environment settings
        self.env_name = 'PongNoFrameskip-v4'
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training settings
        self.num_env_steps = 10_000_000
        self.num_processes = 1
        self.gamma = 0.99
        self.log_interval = 10
        self.save_interval = 100
        
        # Optimization
        self.lr = 2.5e-4
        self.eps = 1e-5
        self.alpha = 0.99
        self.max_grad_norm = 0.5
        
        # Frame processing
        self.frame_stack = 4
        self.frame_skip = 4
        
        # Directories
        self.save_dir = './saved_models'
        self.log_dir = './logs'

class DQNConfig(Config):
    def __init__(self):
        super().__init__()
        self.algorithm = 'DQN'
        self.batch_size = 32
        self.buffer_size = 100_000
        self.learning_starts = 10_000
        self.train_freq = 4
        self.target_update_freq = 1_000
        
        # Epsilon-greedy
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 100_000
        
        # Double DQN
        self.double_dqn = True

class PPOConfig(Config):
    def __init__(self):
        super().__init__()
        self.algorithm = 'PPO'
        self.num_processes = 8
        self.num_steps = 128
        self.batch_size = 32
        self.ppo_epoch = 4
        self.clip_param = 0.1
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.use_gae = True
        self.gae_lambda = 0.95
        self.use_proper_time_limits = False

class A3CConfig(Config):
    def __init__(self):
        super().__init__()
        self.algorithm = 'A3C'
        self.num_processes = 16
        self.num_steps = 20
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.shared_optimizer = True
        self.tau = 1.0

class RainbowConfig(DQNConfig):
    def __init__(self):
        super().__init__()
        self.algorithm = 'Rainbow'
        
        # Prioritized replay
        self.prioritized_replay = True
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_steps = 100_000
        
        # Multi-step returns
        self.n_step = 3
        
        # Distributional RL
        self.distributional = True
        self.v_min = -10.0
        self.v_max = 10.0
        self.atom_size = 51
        
        # Noisy networks
        self.noisy_nets = True
        self.sigma_init = 0.5
        
        # Dueling networks
        self.dueling = True