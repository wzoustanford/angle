# NGU Test Experiment Results

**Date:** 2025-08-06 23:48:50
**Episodes per algorithm:** 6
**Games tested:** ALE/Alien-v5

## Alien Results

| Algorithm | Success | Avg Reward | Time (s) | Notes |
|-----------|---------|------------|----------|-------|
| Baseline DQN | ✓ | 200.00 | 21.9 |  |
| R2D2 | ✗ | 0.00 | 0.4 | 'SequenceReplayBuffer' object has no attribute 'push' |
| Distributed DQN | ✗ | 0.00 | 8.0 | 'DistributedDQNAgent' object has no attribute 'devmgr' |
| NGU | ✓ | 165.00 | 29.0 | Intrinsic: 2723.731 |
| Agent57 | ✓ | 186.67 | 35.3 | Intrinsic: 4757.522 Policies: [0, 1, 2, 3, 4, 5] |
| R2D2+Agent57 | ✗ | 0.00 | 6.1 | NGUNetwork.forward_single_step() got an unexpected keyword argument 'policy_id' |

