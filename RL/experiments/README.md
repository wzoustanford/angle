# RL Experiments

This directory contains experimental scripts for systematic comparison of different RL algorithms and techniques.

## Available Experiments

### Atari Algorithm Comparison (`atari_comparison_experiment.py`)

Comprehensive comparison of four DQN variants across three Atari games:

**Algorithms Tested:**
1. **Basic DQN** - Standard DQN with Double Q-learning
2. **DQN + Priority** - DQN + Double Q + Prioritized Experience Replay
3. **DQN + Priority + Distributed** - Above + Distributed RL (multiple workers)
4. **DQN + Priority + Distributed + R2D2** - Full implementation with LSTM sequences

**Games Tested:**
- Breakout
- SpaceInvaders  
- Pong

**Usage:**

```bash
# Quick test (5 episodes per algorithm)
python experiments/atari_comparison_experiment.py --mode test

# Quick experiment (50 episodes per algorithm)
python experiments/atari_comparison_experiment.py --mode quick

# Full experiment (200 episodes per algorithm - recommended)
python experiments/atari_comparison_experiment.py --mode full

# Custom configuration
python experiments/atari_comparison_experiment.py --episodes 100 --workers 8
```

**Output:**
- Three comparison plots (one per game)
- Combined overview plot
- Detailed results summary
- JSON data export for further analysis

**Expected Runtime:**
- Test mode: ~15 minutes
- Quick mode: ~2-3 hours  
- Full mode: ~8-12 hours (depending on hardware)

## Experiment Design

### Algorithm Configurations

**Basic DQN:**
- Standard replay buffer (uniform sampling)
- Double Q-learning
- Single-threaded training

**DQN + Priority:**
- Prioritized Experience Replay (α=0.6, β=0.4→1.0)
- TD-error based priorities
- Single-threaded training

**DQN + Priority + Distributed:**
- Prioritized replay with 4 parallel workers
- Larger replay buffer (100K vs 50K)
- Larger batch size (64 vs 32)

**DQN + Priority + Distributed + R2D2:**
- LSTM network (512 hidden units)
- Sequence-based learning (80-step sequences)
- Burn-in phase (40 steps)
- All above optimizations

### Training Parameters

| Parameter | Basic/Priority | Distributed | R2D2 |
|-----------|----------------|-------------|------|
| Episodes | 200 | 200 | 200 |
| Buffer Size | 50,000 | 100,000 | 100,000 |
| Batch Size | 32 | 64 | 64 |
| Workers | 1 | 4 | 4 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Target Update | 1000 | 1000 | 1000 |

## Results Analysis

The experiment generates several types of analysis:

1. **Training Curves**: Episode reward vs training episode
2. **Moving Averages**: Smoothed performance trends
3. **Final Performance**: Average of last 20 episodes
4. **Peak Performance**: Best episode reward achieved
5. **Algorithm Ranking**: Ordered by final performance

## Expected Outcomes

Based on RL literature, expected performance ranking:

1. **R2D2 + Distributed + Priority** - Best overall (memory + parallelism + prioritization)
2. **Distributed + Priority** - Strong performance (parallelism + prioritization)  
3. **Priority Replay** - Moderate improvement (better sample efficiency)
4. **Basic DQN** - Baseline performance

**Note**: Results may vary due to:
- Random initialization
- Environment stochasticity  
- Hyperparameter sensitivity
- Training duration

## Hardware Recommendations

**Minimum:**
- 8GB RAM
- 4 CPU cores
- GPU recommended but not required

**Recommended:**
- 16GB+ RAM  
- 8+ CPU cores
- GPU with 4GB+ VRAM
- SSD storage

## Troubleshooting

**Memory Issues:**
- Reduce `--workers` parameter
- Use `--mode quick` instead of `full`
- Close other applications

**Performance Issues:**
- Ensure GPU is available and being used
- Check CPU core count matches `--workers`
- Monitor system resources during training

**Crashes:**
- Try `--mode test` first to verify setup
- Check available disk space for results
- Ensure all dependencies are installed

## Customization

The experiment can be easily extended:

```python
# Add new games
experiment.games = ['ALE/Breakout-v5', 'ALE/Pong-v5', 'ALE/Asteroids-v5']

# Modify training length
experiment.episodes = 500

# Adjust distributed workers
experiment.num_workers = 8

# Add new algorithms (implement config method)
def create_new_algorithm_config(self, env_name):
    # Custom configuration
    pass
```

## Citation

If you use these experiments in research, please cite the relevant papers:

- DQN: Mnih et al. (Nature 2015)
- Double DQN: van Hasselt et al. (AAAI 2016)  
- Prioritized Replay: Schaul et al. (ICLR 2016)
- R2D2: Kapturowski et al. (ICLR 2019)