# RL Examples

This directory contains general RL examples and demonstrations for the distributed reinforcement learning system.

## Available Examples

### 1. Distributed Space Invaders Example (`distributed_space_invaders_example.py`)

Demonstrates the distributed DQN implementation with multiple parallel environments for faster data collection.

**Features:**
- Distributed training with configurable number of workers
- Performance comparison between single-threaded and distributed approaches
- Comprehensive training statistics and visualization
- Multiple training modes (continuous and batch-based)

**Usage:**
```bash
# Basic distributed training
python distributed_space_invaders_example.py --mode train --episodes 200 --workers 4

# Compare single-threaded vs distributed performance
python distributed_space_invaders_example.py --mode compare --episodes 100

# Run system test
python distributed_space_invaders_example.py --mode test

# Help
python distributed_space_invaders_example.py --help
```

**Command-line Options:**
- `--mode`: Choose between 'train', 'compare', or 'test'
- `--episodes`: Number of episodes to train (default: 200)
- `--workers`: Number of parallel workers (default: 4)
- `--no-plot`: Skip plotting results

### 2. Single-Threaded Example (`single_threaded_example.py`)

Traditional single-threaded DQN training approach for comparison and baseline performance.

**Features:**
- Standard DQN implementation
- Training progress visualization
- Performance benchmarking
- Simple configuration

**Usage:**
```bash
# Run single-threaded training
python single_threaded_example.py --episodes 200

# Test setup only
python single_threaded_example.py --test-only

# Skip plotting
python single_threaded_example.py --episodes 100 --no-plot

# Help
python single_threaded_example.py --help
```

### 3. Priority Replay Example (`priority_replay_example_usage.py`)

Demonstrates the configuration and usage of prioritized experience replay vs standard uniform replay.

**Features:**
- Configuration examples for different priority types
- Comparison between uniform and prioritized sampling
- Practical training code examples
- Parameter explanations and recommendations

**Usage:**
```bash
# Show configuration examples
python priority_replay_example_usage.py --mode examples

# Test that prioritized replay works
python priority_replay_example_usage.py --mode test

# Help
python priority_replay_example_usage.py --help
```

**Priority Types Available:**
- `td_error`: Prioritize experiences with high TD-errors (recommended)
- `reward`: Prioritize experiences with high absolute rewards
- `random`: Random priorities (for comparison with uniform sampling)

## Quick Start

1. **Test the system:**
   ```bash
   python distributed_space_invaders_example.py --mode test
   python single_threaded_example.py --test-only
   python priority_replay_example_usage.py --mode test
   ```

2. **Learn about prioritized replay:**
   ```bash
   python priority_replay_example_usage.py --mode examples
   ```

3. **Compare approaches:**
   ```bash
   python distributed_space_invaders_example.py --mode compare --episodes 50
   ```

4. **Run full distributed training:**
   ```bash
   python distributed_space_invaders_example.py --mode train --episodes 200 --workers 4
   ```

## Using the Example Launcher

For convenience, you can use the example launcher from the RL folder:

```bash
# List all available examples
python run_example.py list

# Run specific examples
python run_example.py distributed --args '--mode test'
python run_example.py single --args '--test-only'
python run_example.py priority --args '--mode test'
python run_example.py priority --args '--mode examples'
```

## Key Benefits of Distributed Approach

- **Faster Data Collection**: Multiple environments collect experiences simultaneously
- **Better Sample Efficiency**: Diverse experiences from multiple initializations
- **Scalable Performance**: Easily adjust number of workers based on available CPU cores
- **Improved Training Stability**: More diverse experience replay buffer

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium with ALE support
- NumPy
- Matplotlib (for plotting)
- All dependencies from the main RL package

## Output Files

Examples save results to:
- `../results/distributed_training.png` - Distributed training plots
- `../results/single_threaded_training.png` - Single-threaded training plots
- `../results/distributed_checkpoints/` - Distributed model checkpoints
- `../results/single_threaded_checkpoints/` - Single-threaded model checkpoints

## Customization

Both examples can be easily customized for different Atari games by modifying the configuration:

```python
# For different games
config = DistributedAgentConfig(
    env_name='ALE/Breakout-v5',  # or 'ALE/Pong-v5', etc.
    num_workers=8,
    memory_size=100000
)
```

## Performance Tips

1. **Number of Workers**: Start with number of CPU cores, adjust based on performance
2. **Buffer Size**: Larger buffers for distributed training (50k+ vs 10k for single-threaded)
3. **Batch Size**: Use larger batches for distributed training (64+ vs 32 for single-threaded)
4. **Memory**: Monitor RAM usage with many workers and large buffers

## Troubleshooting

- **ImportError**: Make sure you're running from the examples directory or that the parent directory is in your Python path
- **CUDA Issues**: Examples will automatically use CPU if CUDA is not available
- **Memory Issues**: Reduce number of workers or buffer size if running out of RAM
- **Slow Performance**: Ensure you have sufficient CPU cores for the number of workers