# NGU Test Experiment Results

**Date:** 2025-08-06 23:01:03
**Episodes per algorithm:** 2
**Games tested:** ALE/Alien-v5

## Alien Results

| Algorithm | Success | Avg Reward | Time (s) | Notes |
|-----------|---------|------------|----------|-------|
| Baseline DQN | ✓ | 160.00 | 5.1 |  |
| NGU | ✗ | 0.00 | 2.1 | Expected more than 1 value per channel when training, got input size torch.Size([1, 256]) |
| Agent57 | ✗ | 0.00 | 6.0 | Expected more than 1 value per channel when training, got input size torch.Size([1, 256]) |

