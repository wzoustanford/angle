# NGU Test Experiment Results

**Date:** 2025-08-06 22:50:40
**Episodes per algorithm:** 2
**Games tested:** ALE/Alien-v5

## Alien Results

| Algorithm | Success | Avg Reward | Time (s) | Notes |
|-----------|---------|------------|----------|-------|
| Baseline DQN | ✓ | 160.00 | 5.7 |  |
| NGU | ✗ | 0.00 | 2.1 | SequenceReplayBuffer.__init__() got an unexpected keyword argument 'use_prioritized_replay' |
| Agent57 | ✗ | 0.00 | 2.1 | SequenceReplayBuffer.__init__() got an unexpected keyword argument 'use_prioritized_replay' |

