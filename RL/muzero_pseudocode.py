# Neural Network Models
class MuZeroNetworks:
    def representation(observation):
        """
        h: observation -> hidden_state
        Maps real observation to initial hidden state
        """
        return neural_network_h(observation)
    
    def dynamics(hidden_state, action):
        """
        g: (hidden_state, action) -> (next_hidden_state, reward)
        Predicts next state and reward in latent space
        """
        return neural_network_g(hidden_state, action)
    
    def prediction(hidden_state):
        """
        f: hidden_state -> (policy, value)
        Outputs action probabilities and position evaluation
        """
        return neural_network_f(hidden_state)

# MCTS Planning in Latent Space
def muzero_mcts(observation, num_simulations):
    """
    Performs Monte Carlo Tree Search using learned model
    """
    # Get initial hidden state
    root_state = representation(observation)
    root = Node(state=root_state)
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        
        # SELECTION: Traverse tree using UCB
        while node.is_expanded():
            action, node = select_child(node)  # UCB selection
            search_path.append(node)
        
        # EXPANSION: Expand leaf node
        parent = search_path[-2]
        action = search_path[-1].action
        
        # Use dynamics model to get next state
        next_state, reward = dynamics(parent.state, action)
        node.state = next_state
        node.reward = reward
        
        # Use prediction model at leaf
        policy, value = prediction(next_state)
        node.expand(policy)
        
        # BACKUP: Propagate value up the tree
        backup(search_path, value)
    
    # Return visit counts as action probabilities
    return root.child_visits()

# Training Loop
def train_muzero():
    replay_buffer = []
    networks = MuZeroNetworks()
    
    while training:
        # SELF-PLAY: Generate experience
        trajectory = []
        observation = env.reset()
        
        while not done:
            # Plan with MCTS
            action_probs = muzero_mcts(observation, num_simulations=800)
            action = sample(action_probs)
            
            # Execute in environment
            next_obs, reward, done = env.step(action)
            trajectory.append((observation, action, reward, action_probs))
            observation = next_obs
        
        # Store trajectory with final value
        replay_buffer.add(trajectory)
        
        # LEARNING: Update networks
        batch = replay_buffer.sample()
        
        for trajectory in batch:
            for t, (obs, action, reward, search_policy) in enumerate(trajectory):
                # Initial representation
                hidden_state = representation(obs)
                
                # Unroll dynamics for K steps
                total_loss = 0
                for k in range(K):  # K-step unrolling
                    if k == 0:
                        # Prediction at root
                        policy, value = prediction(hidden_state)
                    else:
                        # Dynamics transition
                        hidden_state, predicted_reward = dynamics(hidden_state, trajectory[t+k].action)
                        policy, value = prediction(hidden_state)
                        
                        # Reward prediction loss
                        total_loss += MSE(predicted_reward, trajectory[t+k].reward)
                    
                    # Policy loss (cross-entropy with MCTS policy)
                    total_loss += cross_entropy(policy, trajectory[t+k].search_policy)
                    
                    # Value loss (MSE with bootstrapped return)
                    bootstrap_value = compute_target_value(trajectory, t+k)
                    total_loss += MSE(value, bootstrap_value)
                
                # Gradient update
                optimize(total_loss)

# Key Helper Functions
def select_child(node):
    """
    Select action using UCB formula
    """
    best_ucb = -inf
    best_action = None
    best_child = None
    
    for action, child in node.children.items():
        ucb = child.value() + c * sqrt(log(node.visits) / child.visits)
        if ucb > best_ucb:
            best_ucb = ucb
            best_action = action
            best_child = child
    
    return best_action, best_child

def backup(search_path, value):
    """
    Propagate value estimates up the tree
    """
    for node in reversed(search_path):
        node.visits += 1
        node.value_sum += value
        value = node.reward + gamma * value  # Discounted backup