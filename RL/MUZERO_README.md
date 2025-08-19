# MuZero Implementation for Atari Games

## Overview

This is an implementation of DeepMind's MuZero algorithm, based on the paper ["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., 2019).

MuZero is a model-based reinforcement learning algorithm that learns a model of the environment and uses Monte Carlo Tree Search (MCTS) for planning. Unlike its predecessors (AlphaGo, AlphaZero), MuZero doesn't require knowledge of the environment's rules.

## Key Components

### 1. Neural Networks (`model/muzero_network.py`)

MuZero uses three neural networks:

- **Representation Network (h)**: Encodes observations into latent state representations
  - Input: Raw observation (e.g., game frame)
  - Output: Hidden state representation

- **Dynamics Network (g)**: Predicts next state and reward
  - Input: Current hidden state + action
  - Output: Next hidden state + immediate reward

- **Prediction Network (f)**: Predicts policy and value
  - Input: Hidden state
  - Output: Action probabilities + state value

### 2. Monte Carlo Tree Search (`model/muzero_mcts.py`)

- Performs simulations to select actions
- Uses UCB (Upper Confidence Bound) for exploration
- Adds Dirichlet noise at root for exploration
- Supports batched MCTS for parallel games

### 3. Replay Buffer (`model/muzero_buffer.py`)

- Stores complete game histories
- Supports prioritized replay based on value prediction error
- Creates training targets with n-step returns
- Handles trajectory unrolling for training

### 4. Agent (`model/muzero_agent.py`)

- Orchestrates training and evaluation
- Manages self-play for data generation
- Implements the training loop with loss computation
- Supports checkpointing and model evaluation

## Configuration

The `MuZeroConfig` class contains all hyperparameters:

```python
from config.MuZeroConfig import MuZeroConfig

config = MuZeroConfig()
config.env_name = 'ALE/SpaceInvaders-v5'  # Atari game
config.num_simulations = 50  # MCTS simulations per move
config.batch_size = 128  # Training batch size
config.learning_rate = 0.001  # Learning rate
```

Key hyperparameters:
- `num_simulations`: Number of MCTS simulations (50 for Atari)
- `discount`: Discount factor (0.997)
- `num_unroll_steps`: Steps to unroll during training (5)
- `td_steps`: Steps for n-step returns (10)
- `support_size`: Size for categorical value/reward representation (300)

## Training

### Basic Training

MuZero supports both **episode-based** and **iteration-based** training:

#### Episode-Based Training (Recommended for experiments)
```bash
# Train MuZero for 100 episodes
python train_muzero.py --game SpaceInvaders --episodes 100

# Train with custom settings
python train_muzero.py --game Breakout --episodes 50 --train-steps-per-episode 100

# Quick test with few episodes
python train_muzero.py --game Alien --episodes 20 --simulations 25
```

#### Iteration-Based Training (For long training runs)
```bash
# Train MuZero for 100,000 iterations
python train_muzero.py --game SpaceInvaders --iterations 100000

# Train on other Atari games
python train_muzero.py --game Breakout --iterations 100000
python train_muzero.py --game Alien --iterations 100000
```

### Command Line Options

- `--game`: Atari game name (default: SpaceInvaders)
- `--episodes`: Number of episodes to train (conflicts with --iterations)
- `--iterations`: Number of training iterations (conflicts with --episodes)
- `--train-steps-per-episode`: Training steps per episode (default: 50, only for episode mode)
- `--simulations`: MCTS simulations per move (default: 50)
- `--batch-size`: Training batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use (cpu/cuda)
- `--checkpoint-dir`: Directory for saving checkpoints
- `--load-checkpoint`: Resume from checkpoint
- `--test-only`: Run evaluation only
- `--render`: Render during evaluation

### Training Modes

**Episode-Based Training:**
- Plays one complete game per episode
- Performs fixed training steps after each episode
- Evaluates every 10 episodes
- Checkpoints every 20 episodes
- Good for experiments and comparing algorithms

**Iteration-Based Training:**
- Alternates between self-play and training
- Self-play every 10 iterations
- More granular control over training
- Good for long training runs

### Algorithm Comparison

Compare MuZero with existing algorithms:

```bash
python experiments/muzero_comparison.py --game SpaceInvaders --episodes 20
```

This will compare:
- Basic DQN
- DQN + Dueling Networks
- DQN + Prioritized Replay
- MuZero

## Testing

### Quick Test
```bash
# Minimal test to verify components work
python test_muzero_minimal.py

# More comprehensive test
python test_muzero_quick.py
```

## Algorithm Details

### Training Process

1. **Self-Play**: Generate games using MCTS with current network
2. **Store**: Save game histories in replay buffer
3. **Sample**: Sample batch of positions from buffer
4. **Unroll**: Create targets using n-step returns
5. **Train**: Update networks to minimize prediction errors
6. **Repeat**: Continue self-play with updated network

### Loss Functions

MuZero optimizes three losses simultaneously:
- **Policy Loss**: Cross-entropy between MCTS policy and predicted policy
- **Value Loss**: MSE/Cross-entropy between n-step return and predicted value
- **Reward Loss**: MSE/Cross-entropy between observed and predicted rewards

### Key Innovations

1. **Learned Model**: Learns dynamics in latent space (no need for rules)
2. **Planning**: Uses MCTS with learned model for decision making
3. **Categorical Representation**: Uses categorical distributions for values/rewards
4. **End-to-end Learning**: Jointly learns representation, dynamics, and planning

## Performance Expectations

On Atari games, MuZero typically:
- Starts learning after ~1000 games
- Shows significant improvement after ~10000 games
- Achieves strong performance after ~100000 games

Training time depends on:
- Number of MCTS simulations (more = better but slower)
- GPU availability (speeds up network training)
- Game complexity (some games are harder to learn)

## File Structure

```
model/
├── muzero_network.py      # Neural network architectures
├── muzero_mcts.py         # Monte Carlo Tree Search
├── muzero_buffer.py       # Replay buffer and game history
└── muzero_agent.py        # Main agent implementation

config/
└── MuZeroConfig.py        # Configuration and hyperparameters

experiments/
└── muzero_comparison.py   # Compare with other algorithms

train_muzero.py            # Main training script
test_muzero_minimal.py     # Minimal functionality test
test_muzero_quick.py       # Quick verification test
```

## References

- [MuZero Paper](https://arxiv.org/abs/1911.08265)
- [DeepMind's Pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
- [MuZero General (Community Implementation)](https://github.com/werner-duvaud/muzero-general)

## Notes

- This implementation follows DeepMind's official pseudocode
- Designed for educational and research purposes
- Optimized for clarity over maximum performance
- Compatible with Gymnasium's Atari environments