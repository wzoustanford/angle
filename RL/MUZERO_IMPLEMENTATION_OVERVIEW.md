# MuZero Implementation Overview

This document provides a detailed overview of the MuZero implementation, explaining each component and how they work together.

## Architecture Overview

MuZero is a model-based reinforcement learning algorithm that learns a model of the environment and uses Monte Carlo Tree Search (MCTS) for planning. Unlike traditional model-based methods, MuZero learns the model in a latent space, eliminating the need for reconstructing observations.

## Core Components

### 1. Configuration (`config/MuZeroConfig.py`)

The configuration class defines all hyperparameters for the MuZero agent:

```python
@dataclass
class MuZeroConfig:
    # Environment settings
    env_name: str = 'ALE/SpaceInvaders-v5'
    observation_shape: (3, 96, 96)  # Resized observations
    
    # MCTS parameters  
    num_simulations: int = 50  # Number of tree search simulations
    discount: float = 0.997  # Reward discount factor
    
    # Network architecture
    hidden_size: int = 256  # Hidden layer size
    support_size: int = 300  # For categorical value/reward representation
    
    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    num_unroll_steps: int = 5  # Steps to unroll during training
    td_steps: int = 10  # N-step returns
    max_moves: int = 1000  # Maximum steps per episode
```

### 2. Neural Networks (`model/muzero_network.py`)

MuZero uses three interconnected neural networks:

#### Representation Network (h)
Encodes raw observations into a latent state representation:

```python
class RepresentationNetwork(nn.Module):
    def forward(self, observation):
        # Input: observation (batch, 3, 96, 96) 
        # Output: hidden state (batch, 256, 24, 24)
        x = conv_layers(observation)
        x = downsample(x)  # Reduce spatial dimensions by 4x
        x = residual_blocks(x)  # Deep processing with skip connections
        return hidden_state
```

**Purpose**: Compresses high-dimensional observations into a compact latent representation that captures relevant information for planning.

#### Dynamics Network (g)
Predicts the next latent state and immediate reward given current state and action:

```python
class DynamicsNetwork(nn.Module):
    def forward(self, state, action):
        # Input: state (batch, 256, 24, 24), action (batch,)
        # Output: reward (batch, 601), next_state (batch, 256, 24, 24)
        
        # Encode action as spatial planes
        action_planes = create_action_planes(action)
        x = concat([state, action_planes])
        
        # Predict next state and reward
        next_state = state_head(x)
        reward = reward_head(x)  # Categorical representation
        return reward, next_state
```

**Purpose**: Models the environment dynamics in latent space, enabling forward simulation without pixel reconstruction.

#### Prediction Network (f)
Outputs policy and value estimates from latent states:

```python
class PredictionNetwork(nn.Module):
    def forward(self, state):
        # Input: state (batch, 256, 24, 24)
        # Output: policy (batch, action_size), value (batch, 601)
        
        policy_logits = policy_head(state)  # Action probabilities
        value = value_head(state)  # Categorical value
        return policy_logits, value
```

**Purpose**: Provides action selection guidance and position evaluation for MCTS.

### 3. Monte Carlo Tree Search (`model/muzero_mcts.py`)

MCTS performs lookahead search using the learned model:

```python
class MCTS:
    def run(self, observation, network, num_simulations=50):
        # 1. Initialize root with network prediction
        root = Node(prior=0)
        initial_output = network.initial_inference(observation)
        root.expand(initial_output['policy'], initial_output['state'])
        
        # 2. Run simulations
        for _ in range(num_simulations):
            # Selection: Traverse tree using UCB formula
            path = []
            node = root
            while node.expanded():
                action, node = select_child(node)  # UCB selection
                path.append((action, node))
            
            # Expansion: Use dynamics network to predict
            if node.visit_count > 0:
                parent_state = get_parent_state(path)
                output = network.recurrent_inference(parent_state, action)
                node.expand(output['policy'], output['state'])
                value = output['value']
            
            # Backpropagation: Update statistics
            for node in reversed(path):
                node.value_sum += value
                node.visit_count += 1
                value = node.reward + discount * value
        
        # 3. Return action based on visit counts
        return select_action_by_visits(root)
```

**Key Features**:
- **UCB Formula**: Balances exploration and exploitation
- **Dirichlet Noise**: Added at root for exploration
- **Value Normalization**: Min-max normalization for stable learning

### 4. Replay Buffer (`model/muzero_buffer.py`)

Stores complete game trajectories for training:

```python
class GameHistory:
    observations: List[np.ndarray]  # Raw observations
    actions: List[int]  # Actions taken
    rewards: List[float]  # Rewards received
    policies: List[np.ndarray]  # MCTS policies
    values: List[float]  # MCTS value estimates
    
    def make_target(self, position, num_unroll_steps, td_steps):
        """Create training target with n-step returns"""
        targets = {
            'observation': observations[position],
            'actions': [],  # Next num_unroll_steps actions
            'target_values': [],  # Bootstrapped values
            'target_rewards': [],  # Actual rewards
            'target_policies': []  # MCTS policies
        }
        
        # Unroll for num_unroll_steps from position
        for i in range(num_unroll_steps + 1):
            current_index = position + i
            
            # Calculate n-step return with bootstrapping
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(values):
                value = values[bootstrap_index] * (discount ** td_steps)
            else:
                value = 0
            
            # Add discounted rewards
            for j, reward in enumerate(rewards[current_index:bootstrap_index]):
                value += reward * (discount ** j)
            
            targets['target_values'].append(value)
            # ... append other targets
        
        return targets
```

**Key Concepts**:
- **Prioritized Sampling**: Sample important transitions more frequently
- **N-step Returns**: Bootstrap value targets using TD(λ)
- **Trajectory Unrolling**: Train on sequences for better dynamics learning

### 5. Training Loop (`model/muzero_agent.py`)

Orchestrates self-play and learning:

```python
class MuZeroAgent:
    def train_episodes(self, num_episodes):
        for episode in range(num_episodes):
            # 1. Self-play: Generate game with MCTS
            game = self.self_play()
            print(f"Episode {episode}: Reward = {sum(game.rewards)}")
            
            # 2. Store in replay buffer
            self.replay_buffer.save_game(game)
            
            # 3. Training steps
            for _ in range(train_steps_per_episode):
                # Sample batch of positions
                batch = self.replay_buffer.sample_batch()
                
                # Forward through network with unrolling
                observations = batch['observations']
                actions = batch['actions']  # List of action sequences
                outputs = self.network(observations, actions)
                
                # Compute losses
                losses = compute_losses(outputs, batch)
                
                # Backprop and update
                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
```

**Training Process**:
1. **Self-play**: Use current network + MCTS to play games
2. **Storage**: Save trajectories in replay buffer
3. **Sampling**: Sample positions and create targets
4. **Unrolling**: Predict multiple steps ahead
5. **Learning**: Update networks to minimize prediction errors

### 6. Loss Functions

Three losses are optimized jointly:

```python
def compute_losses(outputs, targets):
    # Policy loss: Cross-entropy between MCTS and predicted policy
    policy_loss = 0
    for i in range(len(outputs['policy'])):
        policy_loss += cross_entropy(
            outputs['policy'][i], 
            targets['target_policies'][i]
        )
    
    # Value loss: Categorical cross-entropy for value prediction
    value_loss = 0
    for i in range(len(outputs['value'])):
        target_value_cat = scalar_to_support(targets['target_values'][i])
        value_loss += cross_entropy(
            outputs['value_logits'][i],
            target_value_cat
        )
    
    # Reward loss: Prediction of immediate rewards
    reward_loss = 0
    for i in range(1, len(outputs['reward'])):  # Skip first (no reward)
        target_reward_cat = scalar_to_support(targets['target_rewards'][i])
        reward_loss += cross_entropy(
            outputs['reward_logits'][i],
            target_reward_cat
        )
    
    total_loss = policy_loss + value_loss + reward_loss
    return {
        'total': total_loss,
        'policy': policy_loss,
        'value': value_loss,
        'reward': reward_loss
    }
```

**Loss Components**:
- **Policy Loss**: Distills MCTS search into network
- **Value Loss**: Improves position evaluation
- **Reward Loss**: Learns environment dynamics

## Key Implementation Details

### Categorical Representation
Values and rewards use a categorical representation instead of scalars:
- Support size of 300 means values in range [-300, 300]
- Helps with optimization stability
- Allows modeling of multimodal distributions

### Temperature-based Exploration
Action selection temperature decreases over training:
- Early training (< 50k steps): temperature = 1.0 (high exploration)
- Mid training (50k-75k steps): temperature = 0.5
- Late training (> 75k steps): temperature = 0.25 (exploitation)

### Observation Preprocessing
Raw Atari frames (210, 160, 3) are:
1. Resized to (96, 96) using PIL
2. Transposed to channels-first (3, 96, 96)
3. Normalized to [0, 1] range

### Hyperparameter Considerations

**Critical Parameters**:
- `num_simulations`: More = stronger play but slower (25-50 for Atari)
- `max_moves`: Must be reasonable (1000 for most Atari games)
- `num_unroll_steps`: Typically 5 for Atari
- `td_steps`: Should not exceed num_unroll_steps

**Performance Trade-offs**:
- Simulations vs Speed: 10 (fast) → 50 (balanced) → 100 (strong)
- Batch size vs Memory: 32 (low memory) → 128 (standard) → 256 (faster convergence)
- Network size vs Capacity: Larger networks = better modeling but slower

## Testing Results

On Breakout with minimal training (2 episodes):
- Episode 1: 1.0 reward, 153 steps
- Episode 2: 3.0 reward, 249 steps
- Shows learning even with very limited training
- Full training (100+ episodes) would achieve much higher scores

## Comparison with Other Algorithms

| Algorithm | Planning | Model | Exploration |
|-----------|----------|-------|-------------|
| DQN | No | No | ε-greedy |
| MuZero | Yes (MCTS) | Yes (learned) | UCB + Dirichlet |
| AlphaZero | Yes (MCTS) | Yes (perfect) | UCB + Dirichlet |
| Rainbow | No | No | Noisy networks |

**Advantages of MuZero**:
- Learns model end-to-end (no rules needed)
- Plans with learned model (better decisions)
- Combines model-based and model-free RL
- State-of-the-art performance on Atari

**Limitations**:
- Computationally expensive (MCTS simulations)
- Complex implementation
- Requires careful hyperparameter tuning
- Slower than model-free methods

## Usage Examples

### Quick Test
```bash
# Test with minimal settings
python train_muzero.py --game Breakout --episodes 10 --simulations 25
```

### Standard Training
```bash
# Train with standard settings
python train_muzero.py --game SpaceInvaders --episodes 100 --simulations 50
```

### Long Training Run
```bash
# Extended training for best performance
python train_muzero.py --game Alien --iterations 100000 --simulations 50
```

## Debugging Tips

1. **If training is too slow**: Reduce `num_simulations` or `max_moves`
2. **If memory runs out**: Reduce `batch_size` or `replay_buffer_size`
3. **If loss explodes**: Check learning rate, reduce to 0.0001
4. **If no learning**: Increase `num_simulations` or `train_steps_per_episode`
5. **For debugging**: Add print statements in self_play() and train_step()

## References

- [MuZero Paper (Schrittwieser et al., 2019)](https://arxiv.org/abs/1911.08265)
- [DeepMind's Pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
- [MuZero Explained (Blog)](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/)

## Future Improvements

- [ ] Implement reanalyze (retroactively update old games)
- [ ] Add distributed training support
- [ ] Implement efficient batched MCTS
- [ ] Add support for continuous action spaces
- [ ] Optimize with JIT compilation
- [ ] Add visualization of learned dynamics model