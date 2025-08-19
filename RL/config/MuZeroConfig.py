from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MuZeroConfig:
    """Configuration for MuZero agent"""
    
    # Environment
    env_name: str = 'ALE/SpaceInvaders-v5'
    frame_stack: int = 4
    
    # Observation and action spaces
    observation_shape: Tuple[int, int, int] = (3, 96, 96)  # Resized observation
    action_space_size: int = 18  # Standard Atari action space
    
    # MuZero specific parameters
    num_simulations: int = 50  # Number of MCTS simulations per move
    discount: float = 0.997  # Discount factor for rewards
    
    # Network architecture
    representation_size: int = 256  # Size of latent representation
    hidden_size: int = 256  # Hidden layer size for networks
    
    # Dynamics network
    support_size: int = 300  # Support for value/reward transformation
    
    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Replay buffer
    replay_buffer_size: int = 100000  # Number of self-play games to store
    num_unroll_steps: int = 5  # Number of steps to unroll for training
    td_steps: int = 10  # Number of steps for n-step returns
    
    # Self-play
    num_actors: int = 1  # Number of self-play actors
    max_moves: int = 1000  # Maximum moves per game (reasonable for most Atari games)
    num_sampling_moves: int = 30  # Moves with temperature-based exploration
    
    # Root exploration noise
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.5  # Increased from 0.25 for more exploration
    
    # UCB formula constants
    pb_c_base: float = 19652  # Base constant for UCB
    pb_c_init: float = 1.25  # Init constant for UCB
    
    # Training schedule
    training_steps: int = 1000000  # Total training steps
    checkpoint_interval: int = 10000  # Steps between checkpoints
    
    # Temperature schedule for action selection
    visit_softmax_temperature_fn = lambda self, training_steps: 1.0 if training_steps < 50000 else 0.5 if training_steps < 75000 else 0.25
    
    # Device settings
    device: Optional[str] = None  # None = auto-select, 'cpu', 'cuda', 'cuda:0', etc.
    
    # Saving
    checkpoint_dir: str = './results/muzero_checkpoints'
    
    # Categorical representation for values and rewards
    use_categorical: bool = True  # Use categorical representation (as in paper)
    
    # Reanalyze (optional improvement from paper)
    use_reanalyze: bool = False  # Reanalyze old games with latest model
    reanalyze_ratio: float = 0.0  # Ratio of reanalyzed games in batch
    
    # Priority sampling
    use_priority_replay: bool = True
    priority_alpha: float = 1.0  # Priority exponent
    priority_beta: float = 1.0  # Importance sampling exponent
    
    # Debugging and monitoring
    log_interval: int = 100  # Steps between logging
    test_interval: int = 10000  # Steps between test evaluations
    test_episodes: int = 10  # Number of test episodes
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        # Ensure td_steps doesn't exceed num_unroll_steps
        if self.td_steps > self.num_unroll_steps:
            self.td_steps = self.num_unroll_steps