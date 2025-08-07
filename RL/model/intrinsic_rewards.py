import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .episodic_memory import EpisodicMemoryManager
from .random_network_distillation import RNDModule
from .embedding_network import StateEmbeddingNetwork


class NGUIntrinsicReward:
    """
    Never Give Up (NGU) Intrinsic Reward Module
    
    Combines two types of curiosity:
    1. Episodic novelty - from episodic memory (short-term, per-episode)
    2. Lifelong novelty - from Random Network Distillation (long-term, across episodes)
    
    The final intrinsic reward is the product of these two components,
    encouraging exploration of states that are both novel in the current episode
    and novel across the agent's lifetime.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 embedding_dim: int = 128,
                 rnd_feature_dim: int = 512,
                 memory_size: int = 50000,
                 k_neighbors: int = 10,
                 device: torch.device = None):
        """
        Initialize NGU intrinsic reward module
        
        Args:
            input_shape: Input observation shape (C, H, W)
            embedding_dim: Embedding dimension for episodic memory
            rnd_feature_dim: Feature dimension for RND
            memory_size: Maximum episodic memory size
            k_neighbors: Number of neighbors for pseudo-count
            device: Device to run on
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        
        # Embedding network for episodic memory
        self.embedding_network = StateEmbeddingNetwork(
            input_shape=input_shape,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        # Episodic memory manager
        self.episodic_memory = EpisodicMemoryManager(
            embedding_dim=embedding_dim,
            max_memory_size=memory_size,
            k_neighbors=k_neighbors,
            device=self.device
        )
        
        # Random Network Distillation for lifelong novelty
        self.rnd_module = RNDModule(
            input_shape=input_shape,
            feature_dim=rnd_feature_dim,
            device=self.device
        )
        
        # NGU hyperparameters
        self.episodic_reward_scale = 1.0
        self.lifelong_reward_scale = 1.0
        self.l2_threshold = 0.008  # Threshold for episodic novelty
        self.cluster_distance = 0.008  # Distance for memory clustering
        
        # Episode management
        self.current_episode_id = 0
        self.step_count = 0
        
        # Statistics
        self.episode_stats = []
        self.recent_episodic_rewards = []
        self.recent_lifelong_rewards = []
    
    def compute_intrinsic_reward(self, 
                                observations: torch.Tensor,
                                episode_id: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute NGU intrinsic reward combining episodic and lifelong novelty
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            episode_id: Optional episode identifier for memory separation
            
        Returns:
            intrinsic_rewards: Combined intrinsic rewards (batch_size,)
            reward_info: Dictionary with reward breakdown and statistics
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        if episode_id is None:
            episode_id = f"episode_{self.current_episode_id}"
        
        # 1. Compute state embeddings for episodic memory
        with torch.no_grad():
            embeddings = self.embedding_network(observations)
        
        # 2. Compute episodic novelty rewards
        episodic_rewards = []
        for i in range(batch_size):
            embedding = embeddings[i]
            
            # Compute intrinsic reward from episodic memory
            episodic_reward = self.episodic_memory.compute_intrinsic_reward(
                embedding, memory_id=episode_id
            )
            episodic_rewards.append(episodic_reward)
            
            # Add embedding to memory for future queries
            self.episodic_memory.add_embedding(embedding, memory_id=episode_id)
        
        episodic_rewards = torch.tensor(episodic_rewards, device=device, dtype=torch.float32)
        
        # 3. Compute lifelong novelty rewards using RND
        lifelong_rewards = self.rnd_module.compute_intrinsic_reward(observations, normalize=True)
        
        # 4. Combine episodic and lifelong rewards (product as in NGU paper)
        # Scale the rewards
        scaled_episodic = self.episodic_reward_scale * episodic_rewards
        scaled_lifelong = self.lifelong_reward_scale * lifelong_rewards
        
        # Final NGU reward is the product
        intrinsic_rewards = scaled_episodic * scaled_lifelong
        
        # 5. Update statistics
        self._update_statistics(episodic_rewards, lifelong_rewards, intrinsic_rewards)
        
        # 6. Prepare info dictionary
        reward_info = {
            'episodic_rewards': episodic_rewards.detach().cpu().numpy(),
            'lifelong_rewards': lifelong_rewards.detach().cpu().numpy(),
            'combined_rewards': intrinsic_rewards.detach().cpu().numpy(),
            'mean_episodic': float(episodic_rewards.mean()),
            'mean_lifelong': float(lifelong_rewards.mean()),
            'mean_combined': float(intrinsic_rewards.mean()),
            'episode_id': episode_id
        }
        
        self.step_count += batch_size
        
        return intrinsic_rewards, reward_info
    
    def update_networks(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Update the trainable networks (RND predictor and optionally embedding network)
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            
        Returns:
            losses: Dictionary of losses
        """
        losses = {}
        
        # Update RND predictor
        rnd_loss = self.rnd_module.update_predictor(observations)
        losses['rnd_loss'] = rnd_loss
        
        # Note: Embedding network can be trained with contrastive loss or
        # other self-supervised objectives, but for simplicity we keep it fixed here
        # In a full implementation, you might want to add embedding network training
        
        return losses
    
    def reset_episode(self, episode_id: Optional[str] = None):
        """
        Reset episode-specific state (typically called at episode start)
        
        Args:
            episode_id: Optional episode identifier
        """
        if episode_id is None:
            self.current_episode_id += 1
            episode_id = f"episode_{self.current_episode_id}"
        
        # Reset episodic memory for this episode
        if episode_id in self.episodic_memory.memories:
            self.episodic_memory.memories[episode_id].reset()
    
    def _update_statistics(self, 
                          episodic_rewards: torch.Tensor,
                          lifelong_rewards: torch.Tensor, 
                          combined_rewards: torch.Tensor):
        """Update internal statistics for monitoring"""
        # Convert to numpy for easier handling
        episodic_np = episodic_rewards.detach().cpu().numpy()
        lifelong_np = lifelong_rewards.detach().cpu().numpy()
        combined_np = combined_rewards.detach().cpu().numpy()
        
        # Update recent rewards (keep last 1000)
        self.recent_episodic_rewards.extend(episodic_np.tolist())
        self.recent_lifelong_rewards.extend(lifelong_np.tolist())
        
        if len(self.recent_episodic_rewards) > 1000:
            self.recent_episodic_rewards = self.recent_episodic_rewards[-1000:]
            self.recent_lifelong_rewards = self.recent_lifelong_rewards[-1000:]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics for monitoring and debugging"""
        stats = {
            'step_count': self.step_count,
            'current_episode_id': self.current_episode_id,
            'num_active_memories': len(self.episodic_memory.memories),
        }
        
        # Recent reward statistics
        if self.recent_episodic_rewards:
            stats.update({
                'recent_episodic_mean': np.mean(self.recent_episodic_rewards),
                'recent_episodic_std': np.std(self.recent_episodic_rewards),
                'recent_lifelong_mean': np.mean(self.recent_lifelong_rewards),
                'recent_lifelong_std': np.std(self.recent_lifelong_rewards),
            })
        
        # Memory statistics
        memory_stats = self.episodic_memory.get_all_stats()
        stats['memory_stats'] = memory_stats
        
        # RND statistics
        rnd_stats = self.rnd_module.get_stats()
        stats['rnd_stats'] = rnd_stats
        
        return stats
    
    def save_state(self) -> Dict:
        """Save complete state for checkpointing"""
        return {
            'embedding_network': self.embedding_network.state_dict(),
            'episodic_memory': self.episodic_memory.save_all_states(),
            'rnd_module': self.rnd_module.save_state(),
            'current_episode_id': self.current_episode_id,
            'step_count': self.step_count,
            'recent_episodic_rewards': self.recent_episodic_rewards[-100:],
            'recent_lifelong_rewards': self.recent_lifelong_rewards[-100:],
            'hyperparams': {
                'episodic_reward_scale': self.episodic_reward_scale,
                'lifelong_reward_scale': self.lifelong_reward_scale,
                'l2_threshold': self.l2_threshold,
                'cluster_distance': self.cluster_distance
            }
        }
    
    def load_state(self, state: Dict):
        """Load complete state from checkpoint"""
        self.embedding_network.load_state_dict(state['embedding_network'])
        self.episodic_memory.load_all_states(state['episodic_memory'])
        self.rnd_module.load_state(state['rnd_module'])
        
        self.current_episode_id = state.get('current_episode_id', 0)
        self.step_count = state.get('step_count', 0)
        self.recent_episodic_rewards = state.get('recent_episodic_rewards', [])
        self.recent_lifelong_rewards = state.get('recent_lifelong_rewards', [])
        
        # Load hyperparameters
        if 'hyperparams' in state:
            hyperparams = state['hyperparams']
            self.episodic_reward_scale = hyperparams.get('episodic_reward_scale', 1.0)
            self.lifelong_reward_scale = hyperparams.get('lifelong_reward_scale', 1.0)
            self.l2_threshold = hyperparams.get('l2_threshold', 0.008)
            self.cluster_distance = hyperparams.get('cluster_distance', 0.008)


class Agent57IntrinsicReward(NGUIntrinsicReward):
    """
    Extended intrinsic reward module for Agent57
    
    Adds meta-learning capabilities with multiple exploration strategies.
    Each meta-policy can have its own episodic memory and exploration parameters.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 num_policies: int = 32,
                 embedding_dim: int = 128,
                 rnd_feature_dim: int = 512,
                 memory_size: int = 50000,
                 device: torch.device = None):
        """
        Initialize Agent57 intrinsic reward module
        
        Args:
            input_shape: Input observation shape (C, H, W)
            num_policies: Number of meta-policies
            embedding_dim: Embedding dimension for episodic memory
            rnd_feature_dim: Feature dimension for RND
            memory_size: Maximum episodic memory size per policy
            device: Device to run on
        """
        super().__init__(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            rnd_feature_dim=rnd_feature_dim,
            memory_size=memory_size,
            device=device
        )
        
        self.num_policies = num_policies
        
        # Policy-specific exploration parameters
        # Beta values control exploration vs exploitation balance
        self.policy_betas = np.linspace(0.0, 0.3, num_policies)  # 0 = pure exploitation, higher = more exploration
        
        # Gamma values for intrinsic reward discounting
        self.policy_gammas = np.linspace(0.99, 0.997, num_policies)  # Different time horizons
        
        # Current active policy
        self.current_policy_id = 0
        
        # Policy-specific statistics
        self.policy_stats = {i: {'episodes': 0, 'steps': 0, 'avg_reward': 0.0} 
                            for i in range(num_policies)}
    
    def set_active_policy(self, policy_id: int):
        """Set the currently active meta-policy"""
        assert 0 <= policy_id < self.num_policies, f"Policy ID must be in [0, {self.num_policies})"
        self.current_policy_id = policy_id
    
    def compute_policy_intrinsic_reward(self, 
                                      observations: torch.Tensor,
                                      policy_id: Optional[int] = None,
                                      episode_id: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute intrinsic reward for specific policy
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            policy_id: Policy ID (uses current if None)
            episode_id: Episode identifier
            
        Returns:
            intrinsic_rewards: Policy-specific intrinsic rewards
            reward_info: Reward breakdown and policy info
        """
        if policy_id is None:
            policy_id = self.current_policy_id
        
        if episode_id is None:
            episode_id = f"policy_{policy_id}_episode_{self.current_episode_id}"
        
        # Get base NGU intrinsic rewards
        intrinsic_rewards, reward_info = self.compute_intrinsic_reward(
            observations, episode_id=episode_id
        )
        
        # Apply policy-specific scaling
        beta = self.policy_betas[policy_id]
        policy_intrinsic_rewards = beta * intrinsic_rewards
        
        # Update reward info with policy details
        reward_info.update({
            'policy_id': policy_id,
            'policy_beta': beta,
            'policy_gamma': self.policy_gammas[policy_id],
            'base_intrinsic_rewards': reward_info['combined_rewards'],
            'policy_scaled_rewards': policy_intrinsic_rewards.detach().cpu().numpy()
        })
        
        # Update policy statistics
        self.policy_stats[policy_id]['steps'] += observations.shape[0]
        
        return policy_intrinsic_rewards, reward_info
    
    def get_policy_statistics(self) -> Dict:
        """Get statistics for all policies"""
        base_stats = self.get_statistics()
        base_stats.update({
            'num_policies': self.num_policies,
            'current_policy_id': self.current_policy_id,
            'policy_betas': self.policy_betas.tolist(),
            'policy_gammas': self.policy_gammas.tolist(),
            'policy_stats': self.policy_stats
        })
        return base_stats


def test_ngu_intrinsic_reward():
    """Test function for NGU intrinsic reward"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing NGU Intrinsic Reward on device: {device}")
    
    # Test parameters
    batch_size = 4
    input_shape = (4, 84, 84)  # Atari frame stack
    
    # Create NGU module
    ngu_reward = NGUIntrinsicReward(
        input_shape=input_shape,
        embedding_dim=64,  # Smaller for testing
        rnd_feature_dim=256,  # Smaller for testing
        memory_size=1000,  # Smaller for testing
        device=device
    )
    
    print(f"Created NGU module with embedding_dim=64, rnd_feature_dim=256")
    
    # Generate test observations
    observations = torch.randint(0, 256, (batch_size, *input_shape), 
                                dtype=torch.float32, device=device)
    
    print(f"Input shape: {observations.shape}")
    
    # Test intrinsic reward computation
    intrinsic_rewards, reward_info = ngu_reward.compute_intrinsic_reward(observations)
    
    print(f"Intrinsic rewards: {intrinsic_rewards}")
    print(f"Mean episodic reward: {reward_info['mean_episodic']:.4f}")
    print(f"Mean lifelong reward: {reward_info['mean_lifelong']:.4f}")
    print(f"Mean combined reward: {reward_info['mean_combined']:.4f}")
    
    # Test network updates
    losses = ngu_reward.update_networks(observations)
    print(f"RND loss: {losses['rnd_loss']:.4f}")
    
    # Test with same observations (should have lower novelty)
    intrinsic_rewards2, reward_info2 = ngu_reward.compute_intrinsic_reward(observations)
    print(f"Second time rewards: {intrinsic_rewards2}")
    print(f"Reward change: {(intrinsic_rewards2 - intrinsic_rewards).mean():.4f}")
    
    # Test statistics
    stats = ngu_reward.get_statistics()
    print("\nNGU Statistics:")
    print(f"  Step count: {stats['step_count']}")
    print(f"  Active memories: {stats['num_active_memories']}")
    if 'recent_episodic_mean' in stats:
        print(f"  Recent episodic mean: {stats['recent_episodic_mean']:.4f}")
        print(f"  Recent lifelong mean: {stats['recent_lifelong_mean']:.4f}")
    
    # Test Agent57 version
    print("\n--- Testing Agent57 Intrinsic Reward ---")
    agent57_reward = Agent57IntrinsicReward(
        input_shape=input_shape,
        num_policies=8,  # Smaller for testing
        embedding_dim=64,
        rnd_feature_dim=256,
        memory_size=1000,
        device=device
    )
    
    # Test different policies
    for policy_id in [0, 4, 7]:  # Test different exploration levels
        agent57_reward.set_active_policy(policy_id)
        policy_rewards, policy_info = agent57_reward.compute_policy_intrinsic_reward(observations)
        print(f"Policy {policy_id} (beta={agent57_reward.policy_betas[policy_id]:.3f}): {policy_rewards.mean():.4f}")
    
    print("✓ NGU Intrinsic Reward test passed!")


if __name__ == "__main__":
    test_ngu_intrinsic_reward()