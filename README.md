# Angle.ac 

# Angle MatrixRL: Deep RL for Online and performance 

Welcome to **Angle MatrixRL**, an open-source reinforcement learning framework built for research and real-time system performance. This repo offers a clean, extensible, and reproducible codebase to train agents on **Atari**, **Retro**, and other classic environments using modern deep RL algorithms. 

note, we are working on moving code from priavte repo. So the structure isn't there yet. 
the open source project just started, stay tuned. 
---

## Features

- **RL algorithms towards State-of-the-art**: DQN, PPO, A2C, Rainbow, NEC, Agent57 and more.
- **towards online RL**: Towards industry grade efficiency. Components trusted by real-time production services. 
- **Multi-environment support**: Atari (ALE via Gymnasium), economic and financial data, custom environments.
- **Modular & extensible** design: Easily plug in new agents, models, and environments.
- **Training dashboards** with TensorBoard integration.
- **Replay buffer**, frame stacking, action repeat, priority buffers.
- **LoggingExperiment tracking**: Reproducible configs for clean ablations and comparisons.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/wzoustanford/angle.git
cd angle
pip install -r requirements.txt
```

### 2. Train an agent
```bash
python train.py --env BreakoutNoFrameskip-v4 --algo dqn --config configs/dqn_default.yaml
```

Or try PPO on a Retro game:
```bash
python train.py --env SonicTheHedgehog-Genesis --algo ppo --config configs/ppo_retro.yaml
```

Watch the agent play: 
```bash
python play.py --checkpoint runs/dqn_breakout/best_model.pth --env BreakoutNoFrameskip-v4
```

## Project structure
```graphql 
angle/RL/
│
├── models/         # Neural network architectures
├── DataFeed.py     # Replay buffer, logger, scheduler, etc.
├── AgentConfig.py  # YAML configs for experiments
├── train.py        # Entry point for training [wip]
├── play.py         # Agent evaluation / gameplay rendering
├── agents/         # RL algorithms (DQN, PPO, A2C, etc.) [wip] 
├── envs/           # Atari, Retro, and custom env wrappers [wip]
└── ...
```

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue, suggest improvements, or submit a pull request.

To get started:
```bash
git checkout -b my-feature
# Make awesome changes
git commit -m "Add amazing feature"
git push origin my-feature
```

## -- AgentTrainer 

```python
class AgentTrainer 
  def __init():
    self.replay_buffer = ReplayBuffer()
  def train():
    initial_action 
    initial_state 
    state, reward, data-feed(initial_state, initial_action)
    for t in T: 
        ## -- sample action from policy 
        action = policy_function(state) (produce action)
        
        #data-feed: abstract into the step function, given action and time-step, and current state, give the next state and the reward
        next_state, reward = data-feed(initial_state, action)

        ## -- manage replay buffer 
        self.replay_buffer.insert(state, action, next_state) 
        
        prev_states_tensor, actions_tensor, states_tensor = self.replay_buffer.get_batch(B) 
        
        next_actions_tensor = policy_function.forward_tensor(policy_function, states_tensor)

        ## -- Meta learning 
        if meta_learning_condition: 
            meta_learner_trainer.update_meta_learning_model(states_tensor, Q)
            # e.g. update clinical task: patient clinical presentation, biological results forecasting/monitoring
            meta_learner_trainer.update_substrate_model_clinical_forecasting(states_tensor, Q)
            # e.g. update clinical task: risk detection
            meta_learner_trainer.update_substrate_model_with_risk_detection(states_tensor, Q)

        ## -- Q learning 
        Q_future = Q_future_function(states_tensor, next_actions_tensor) 
        y = reward + gamma * Q_future 
        loss = torch.nn.mse_loss(y, Q(prev_states_tensor, actions_tensor)) 
        
        Q_optimizer.zero_grad() 
        loss.backward()
        Q_optimizer.step()
    
        if t % policy_update_interval == 0: 
            prev_states_tensor, actions_tensor, states_tensor = replay_get_batch(B)
          	next_actions_sampled = policy_function.forward_tensor(policy_function, states_tensor) 
          	q = Q(states_tensor, next_actions_sampled)
          	loss = -1.0 * q
          	policy_optimizer.zero_grad()
          	loss.backward()
          	policy_optimizer.step()
    
# abstractions for policy model 
class PolicyModel(torch.nn.Module):
    def __init__(self):
    def forward(self, x):
    def forward_tensor(self, x): # tensor management 
# abstractions for replay buffer
class ReplayBuffer:

# abstractions for meta-learning 
class MetaLearnerTrainer(AgentTrainer): 

# abstractions for Q-learning 
class QLearning: 
  def Q_future_function(states_tensor, next_actions_sampled):
      return torch.minimum(Q_1_d(states_tensor, next_actions_sampled), Q_2_d(states_tensor, next_actions_sampled))
  def Q_future_function_tensor(): # tensor management 
    
# managing trainer configurations 
AgentConfig - defines configs for hyperparameters, training, service, includes device management 
