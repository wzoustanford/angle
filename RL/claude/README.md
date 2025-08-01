# Atari Reinforcement Learning Suite

A comprehensive PyTorch implementation of state-of-the-art reinforcement learning algorithms for Atari games.

## Implemented Algorithms

1. **DQN (Deep Q-Network)** - Basic DQN with experience replay and target networks
2. **PPO (Proximal Policy Optimization)** - On-policy actor-critic method with clipped surrogate objective
3. **A3C (Asynchronous Advantage Actor-Critic)** - Asynchronous multi-process training
4. **Rainbow DQN** - Combines six improvements to DQN:
   - Double Q-learning
   - Prioritized experience replay
   - Dueling networks
   - Multi-step learning
   - Distributional RL (C51)
   - Noisy networks

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd atari-rl-suite

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py           # Configuration classes for all algorithms
├── networks.py         # Neural network architectures
├── envs.py            # Atari environment wrappers
├── replay_buffer.py   # Experience replay implementations
├── utils.py           # Utility functions
├── dqn.py            # DQN implementation
├── ppo.py            # PPO implementation
├── a3c.py            # A3C implementation
├── rainbow.py        # Rainbow DQN implementation
└── main.py           # Main training/evaluation script
```

## Usage

### Training

```bash
# Train DQN on Pong
python main.py --algorithm dqn --env PongNoFrameskip-v4

# Train PPO on Breakout with custom hyperparameters
python main.py --algorithm ppo --env BreakoutNoFrameskip-v4 --lr 0.0003 --num-processes 8

# Train A3C on Space Invaders
python main.py --algorithm a3c --env SpaceInvadersNoFrameskip-v4 --num-processes 16

# Train Rainbow on Qbert
python main.py --algorithm rainbow --env QbertNoFrameskip-v4
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --algorithm dqn --env PongNoFrameskip-v4 --eval --checkpoint saved_models/dqn_1000000.pth
```

### Command Line Arguments

- `--algorithm`: Choose algorithm (dqn, ppo, a3c, rainbow)
- `--env`: Atari environment name
- `--seed`: Random seed for reproducibility
- `--num-steps`: Total number of training steps
- `--lr`: Learning rate
- `--batch-size`: Batch size for training
- `--num-processes`: Number of parallel processes (PPO/A3C)
- `--device`: Device to use (cuda/cpu)
- `--eval`: Enable evaluation mode
- `--checkpoint`: Path to checkpoint for evaluation

## Algorithm Details

### DQN
- Experience replay buffer
- Target network updated periodically
- ε-greedy exploration
- Optional double DQN

### PPO
- Vectorized environments for parallel collection
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function clipping

### A3C
- Asynchronous multi-process training
- Shared Adam optimizer
- On-policy actor-critic
- Entropy regularization

### Rainbow
- Categorical DQN (C51) for distributional RL
- Prioritized experience replay with importance sampling
- Dueling network architecture
- Multi-step returns
- Noisy networks for exploration
- Double Q-learning

## Environment Preprocessing

All Atari environments use standard preprocessing:
- Frame skipping (4 frames)
- Gray-scale conversion
- Frame resizing to 84x84
- Frame stacking (4 frames)
- Reward clipping to [-1, 1]
- Episode termination on life loss

## Extending the Code

### Adding New Algorithms

1. Create a new config class in `config.py`
2. Implement the algorithm in a new file
3. Add the algorithm to `main.py`

### Adding Agent57

The codebase is designed to be extended with Agent57. To implement it:

1. Add NGU (Never Give Up) intrinsic rewards
2. Implement the meta-controller for β selection
3. Add episodic memory for intrinsic rewards
4. Implement the mixture of policies
5. Add the adaptive exploration schedule

## Performance Tips

- Use GPU acceleration when available (`--device cuda`)
- Adjust number of processes based on CPU cores
- For Rainbow, prioritized replay requires more memory
- A3C benefits from more processes (16-32)
- PPO works well with 8-16 processes

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Reduce number of parallel environments
- Use gradient accumulation

### Slow Training
- Ensure GPU is being used
- Check environment preprocessing isn't bottlenecked
- Use fewer parallel environments if CPU-bound

### Poor Performance
- Ensure proper seed setting for reproducibility
- Check hyperparameters match paper recommendations
- Verify environment wrappers are correctly applied

## Citation

If you use this code, please cite the original papers:
- DQN: Mnih et al., "Human-level control through deep reinforcement learning"
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms"
- A3C: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning"
- Rainbow: Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"

## License

MIT License