# RL Algorithms Comparison Results

**Experiment Date:** 2025-08-06 20:01:02
**Episodes per Algorithm:** 5
**Games Tested:** Alien, IceHockey
**Total Runtime:** 858.0 seconds

## Algorithms Tested

1. **Basic DQN** - Standard Deep Q-Network with Double Q-learning
2. **DQN + Dueling Networks** - DQN with separate value and advantage streams
3. **DQN + Priority Replay** - DQN with prioritized experience replay

## Results Summary

### Alien

| Algorithm | Avg Reward | Best Episode | Worst Episode | Training Time |
|-----------|------------|--------------|---------------|---------------|
| 🥇 DQN + Dueling | 178.00 | 210.0 | 130.0 | 80.9s |
| 🥈 DQN + Priority | 176.00 | 230.0 | 130.0 | 87.0s |
| 🥉 Basic DQN | 152.00 | 250.0 | 80.0 | 78.9s |

#### Alien - Episode Details

| Episode | Basic DQN | DQN + Dueling | DQN + Priority |
|---------|-----------|---------------|----------------|
| 1 | 80.0 | 130.0 | 180.0 |
| 2 | 200.0 | 190.0 | 160.0 |
| 3 | 250.0 | 210.0 | 230.0 |
| 4 | 140.0 | 190.0 | 130.0 |
| 5 | 90.0 | 170.0 | 180.0 |

### IceHockey

| Algorithm | Avg Reward | Best Episode | Worst Episode | Training Time |
|-----------|------------|--------------|---------------|---------------|
| 🥇 DQN + Priority | -3.60 | -3.0 | -4.0 | 203.3s |
| 🥈 DQN + Dueling | -4.40 | -2.0 | -7.0 | 202.5s |
| 🥉 Basic DQN | -4.60 | -4.0 | -6.0 | 201.5s |

#### IceHockey - Episode Details

| Episode | Basic DQN | DQN + Dueling | DQN + Priority |
|---------|-----------|---------------|----------------|
| 1 | -4.0 | -4.0 | -4.0 |
| 2 | -5.0 | -6.0 | -4.0 |
| 3 | -4.0 | -7.0 | -3.0 |
| 4 | -6.0 | -2.0 | -4.0 |
| 5 | -4.0 | -3.0 | -3.0 |

## Key Findings

- **Best Overall Algorithm:** DQN + Dueling (avg score: 86.80)
- **Best for Alien:** DQN + Dueling (178.00 avg reward)
- **Best for IceHockey:** DQN + Priority (-3.60 avg reward)

## Files Generated

- `results.json` - Raw numerical results
- `experiment.log` - Detailed execution log
- `comparison_plot.png` - Main comparison visualization
- `alien_plot.png` - Alien game learning curves
- `icehockey_plot.png` - Ice Hockey game learning curves
