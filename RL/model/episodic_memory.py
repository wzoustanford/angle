import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


class EpisodicMemory:
    """
    Episodic Memory for NGU (Never Give Up) Algorithm
    
    Stores state embeddings and computes pseudo-counts for intrinsic motivation.
    Based on "Never Give Up: Learning Directed Exploration from Human Demonstrations"
    
    Key Features:
    - Efficient nearest neighbor search using approximate methods
    - Controllable memory size with LRU eviction
    - Fast pseudo-count computation for intrinsic rewards
    - Support for multiple embedding dimensions
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 max_memory_size: int = 50000,
                 k_neighbors: int = 10,
                 kernel_epsilon: float = 0.001,
                 cluster_distance: float = 0.008,
                 c_constant: float = 0.001,
                 device: torch.device = None):
        """
        Initialize episodic memory
        
        Args:
            embedding_dim: Dimension of state embeddings
            max_memory_size: Maximum number of stored embeddings
            k_neighbors: Number of neighbors for pseudo-count calculation
            kernel_epsilon: Small constant for kernel similarity
            cluster_distance: Distance threshold for clustering
            c_constant: Constant for pseudo-count calculation
            device: Device to store tensors on
        """
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.k_neighbors = k_neighbors
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.c_constant = c_constant
        self.device = device or torch.device('cpu')
        
        # Memory storage
        self.memory = torch.zeros((max_memory_size, embedding_dim), device=self.device)
        self.memory_count = 0  # Number of stored embeddings
        self.next_index = 0    # Next position to write (LRU)
        
        # Fast similarity computation cache
        self.last_query = None
        self.last_similarities = None
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        
    def add_embedding(self, embedding: torch.Tensor) -> bool:
        """
        Add a new embedding to memory
        
        Args:
            embedding: State embedding vector (embedding_dim,)
            
        Returns:
            bool: True if embedding was added (not too similar to existing ones)
        """
        embedding = embedding.detach().to(self.device)
        
        # Check if this embedding is too similar to existing ones (clustering)
        if self.memory_count > 0:
            distances = torch.norm(self.memory[:self.memory_count] - embedding, dim=1)
            min_distance = torch.min(distances).item()
            
            if min_distance < self.cluster_distance:
                return False  # Too similar, don't add
        
        # Add to memory
        self.memory[self.next_index] = embedding
        self.next_index = (self.next_index + 1) % self.max_memory_size
        
        if self.memory_count < self.max_memory_size:
            self.memory_count += 1
            
        # Invalidate cache since memory changed
        self.last_query = None
        self.last_similarities = None
        
        return True
    
    def compute_pseudo_count(self, query_embedding: torch.Tensor) -> float:
        """
        Compute pseudo-count for a query embedding using k-nearest neighbors
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            
        Returns:
            float: Pseudo-count value (higher = more visited)
        """
        if self.memory_count == 0:
            return 0.0
            
        query_embedding = query_embedding.detach().to(self.device)
        self.total_queries += 1
        
        # Check cache
        if (self.last_query is not None and 
            torch.allclose(query_embedding, self.last_query, atol=1e-6)):
            self.cache_hits += 1
            similarities = self.last_similarities
        else:
            # Compute similarities to all stored embeddings
            # Using RBF kernel: exp(-||x - y||^2 / epsilon)
            distances = torch.norm(
                self.memory[:self.memory_count] - query_embedding.unsqueeze(0), 
                dim=1
            )
            similarities = torch.exp(-distances.pow(2) / self.kernel_epsilon)
            
            # Cache the result
            self.last_query = query_embedding.clone()
            self.last_similarities = similarities
        
        # Get k nearest neighbors
        k = min(self.k_neighbors, self.memory_count)
        top_k_similarities, _ = torch.topk(similarities, k)
        
        # Compute pseudo-count as sum of top-k similarities
        pseudo_count = top_k_similarities.sum().item()
        
        return pseudo_count
    
    def compute_intrinsic_reward(self, query_embedding: torch.Tensor) -> float:
        """
        Compute intrinsic reward based on pseudo-count
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            float: Intrinsic reward (higher for less visited states)
        """
        pseudo_count = self.compute_pseudo_count(query_embedding)
        
        # Intrinsic reward: 1 / sqrt(N + c) where N is pseudo-count
        intrinsic_reward = 1.0 / np.sqrt(pseudo_count + self.c_constant)
        
        return intrinsic_reward
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics for debugging"""
        cache_hit_rate = (self.cache_hits / max(self.total_queries, 1)) * 100
        
        return {
            'memory_count': self.memory_count,
            'memory_utilization': f"{(self.memory_count / self.max_memory_size) * 100:.1f}%",
            'total_queries': self.total_queries,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'next_index': self.next_index,
            'is_full': self.memory_count >= self.max_memory_size
        }
    
    def reset(self):
        """Clear all stored memories"""
        self.memory_count = 0
        self.next_index = 0
        self.last_query = None
        self.last_similarities = None
        self.total_queries = 0
        self.cache_hits = 0
    
    def save_state(self) -> Dict:
        """Save memory state for checkpointing"""
        return {
            'memory': self.memory.cpu(),
            'memory_count': self.memory_count,
            'next_index': self.next_index,
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits
        }
    
    def load_state(self, state: Dict):
        """Load memory state from checkpoint"""
        self.memory = state['memory'].to(self.device)
        self.memory_count = state['memory_count']
        self.next_index = state['next_index']
        self.total_queries = state.get('total_queries', 0)
        self.cache_hits = state.get('cache_hits', 0)
        
        # Invalidate cache
        self.last_query = None
        self.last_similarities = None


class EpisodicMemoryManager:
    """
    Manager for multiple episodic memories (e.g., per environment or per meta-policy)
    Used in Agent57 with multiple exploration strategies
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 max_memory_size: int = 50000,
                 k_neighbors: int = 10,
                 device: torch.device = None):
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.k_neighbors = k_neighbors
        self.device = device or torch.device('cpu')
        
        # Dictionary of memories indexed by policy/environment ID
        self.memories: Dict[str, EpisodicMemory] = {}
    
    def get_memory(self, memory_id: str = "default") -> EpisodicMemory:
        """Get or create an episodic memory for given ID"""
        if memory_id not in self.memories:
            self.memories[memory_id] = EpisodicMemory(
                embedding_dim=self.embedding_dim,
                max_memory_size=self.max_memory_size,
                k_neighbors=self.k_neighbors,
                device=self.device
            )
        return self.memories[memory_id]
    
    def add_embedding(self, embedding: torch.Tensor, memory_id: str = "default") -> bool:
        """Add embedding to specific memory"""
        memory = self.get_memory(memory_id)
        return memory.add_embedding(embedding)
    
    def compute_intrinsic_reward(self, embedding: torch.Tensor, memory_id: str = "default") -> float:
        """Compute intrinsic reward from specific memory"""
        memory = self.get_memory(memory_id)
        return memory.compute_intrinsic_reward(embedding)
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all memories"""
        return {memory_id: memory.get_memory_stats() 
                for memory_id, memory in self.memories.items()}
    
    def reset_all(self):
        """Reset all memories"""
        for memory in self.memories.values():
            memory.reset()
    
    def save_all_states(self) -> Dict[str, Dict]:
        """Save states of all memories"""
        return {memory_id: memory.save_state() 
                for memory_id, memory in self.memories.items()}
    
    def load_all_states(self, states: Dict[str, Dict]):
        """Load states for all memories"""
        for memory_id, state in states.items():
            self.get_memory(memory_id).load_state(state)