import numpy as np


class SumTree:
    """
    Sum tree data structure for efficient priority-based sampling.
    
    This implementation supports O(log n) updates and sampling operations.
    The tree stores priorities in internal nodes and data indices in leaf nodes.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate the change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float):
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return total priority sum"""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add new data with given priority"""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of data at tree index"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float):
        """Get data sample based on priority value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[data_idx])
    
    def sample(self, batch_size: int):
        """Sample batch_size items based on priorities"""
        batch = []
        idxs = []
        priorities = []
        
        segment = self.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            (idx, p, data) = self.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        return batch, idxs, priorities