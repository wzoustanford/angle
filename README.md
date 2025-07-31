# Angle

OpenRL Reinforcement Learning repository 

Abstractions 

AgentConfig - defines configs for hyperparameters, training, service, 

class AgentTrainer 
  def __init()
  def train():
    initial_action
    initial_state 
    state, reward, data-feed(initial_state, initial_action)
    for t in T: 
        action = policy_function(state) (produce action)
        
        #data-feed: abstract into the step function, given action and time-step, and current state, give the next state and the reward
        next_state, reward, data-feed(initial_state, action)
        
        insert_into_replay(state, action, next_state) 
        
        prev_states_tensor, actions_tensor, states_tensor = replay_get_batch(B) 
        
        next_actions_sampled = get_sampled_actions(, policy_function, states_tensor)
        
        Q_future = Q_future_function(states_tensor, next_actions_sampled) 
        y = reward + gamma * Q_future 
        loss = torch.nn.mse_loss(y, Q(prev_states_tensor, actions_tensor)) 
        
        Q_optimizer.zero_grad() 
        loss.backward()
        Q_optimizer.step()
    
        if t % policy_update_interval == 0: 
            prev_states_tensor, actions_tensor, states_tensor = replay_get_batch(B) 
    	next_actions_sampled = get_sampled_actions(policy_function, states_tensor) 
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
# abstractions for meta-learning 
class MetaLearnerTrainer(AgentTrainer): 

# abstractions for Q-learning 
def Q_future_function(states_tensor, next_actions_sampled):
    return torch.minimum(Q_1_d(states_tensor, next_actions_sampled), Q_2_d(states_tensor, next_actions_sampled))
    
