# Angle

OpenRL Reinforcement Learning repository 

# 🕹️ Angle OpenRL: Deep Reinforcement Learning for Atari and Beyond

Welcome to **Angle OpenRL**, an open-source reinforcement learning framework built for fun, research, and serious performance. This repo offers a clean, extensible, and reproducible codebase to train agents on **Atari**, **Retro**, and other classic environments using modern deep RL algorithms.

Whether you're here to beat Breakout, train a Pac-Man prodigy, or prototype your own AI, you’re in the right place.

---

## 🚀 Features

- 🧠 **State-of-the-art RL algorithms**: DQN, PPO, A2C, Rainbow, and more.
- 🎮 **Multi-environment support**: Atari (ALE via Gymnasium), OpenAI Retro, and custom games.
- 🛠️ **Modular & extensible** design: Easily plug in new agents, models, and environments.
- 📈 **Training dashboards** with TensorBoard and Weights & Biases integration.
- ♻️ **Replay buffer**, frame stacking, action repeat, and other training tricks.
- 🧪 **Experiment tracking**: Reproducible configs for clean ablations and comparisons.

---

## 🧰 Quick Start

### 1. Install

```bash
git clone https://github.com/wzoustanford/angle.git
cd angle
pip install -r requirements.txt

# Train an agent
'''bash
python train.py --env BreakoutNoFrameskip-v4 --algo dqn --config configs/dqn_default.yaml

# Project structure
'''graphql 
angle/
│
├── agents/         # RL algorithms (DQN, PPO, A2C, etc.)
├── envs/           # Atari, Retro, and custom env wrappers
├── models/         # Neural network architectures
├── utils/          # Replay buffer, logger, scheduler, etc.
├── configs/        # YAML configs for experiments
├── train.py        # Entry point for training
├── play.py         # Agent evaluation / gameplay rendering
└── ...


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
    
        if t % meta_update_interval == 0:
            meta_learner_trainer.update_meta_learning_model(states_tensor, Q)

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
