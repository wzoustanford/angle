import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StateEmbeddingNetwork(nn.Module):
    """
    State Embedding Network for NGU Episodic Memory
    
    Converts raw observations into fixed-size embeddings for similarity computation
    in episodic memory. The network is designed to learn representations that
    capture semantically meaningful state features.
    
    Architecture:
    - Reuses CNN features from R2D2/DQN for computational efficiency
    - Additional embedding layers to create compact representations
    - L2 normalization for stable similarity computation
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 embedding_dim: int = 128,
                 cnn_feature_dim: int = 512,
                 use_batch_norm: bool = True):
        """
        Initialize state embedding network
        
        Args:
            input_shape: Input observation shape (C, H, W)
            embedding_dim: Output embedding dimension
            cnn_feature_dim: CNN feature dimension (512 for DQN/R2D2 compatibility)
            use_batch_norm: Whether to use batch normalization
        """
        super(StateEmbeddingNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.cnn_feature_dim = cnn_feature_dim
        
        # CNN backbone (compatible with DQN/R2D2)
        self.cnn = self._build_cnn(input_shape)
        
        # Embedding head (avoid BatchNorm for single batch compatibility)
        layers = []
        
        # First projection layer
        layers.append(nn.Linear(cnn_feature_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Second projection layer  
        layers.append(nn.Linear(256, embedding_dim))
        
        self.embedding_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _build_cnn(self, input_shape: Tuple[int, int, int]):
        """Build CNN identical to DQN/R2D2 for feature extraction"""
        c, h, w = input_shape
        
        return nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._calc_conv_output_size(input_shape), 512),
            nn.ReLU()
        )
    
    def _calc_conv_output_size(self, input_shape: Tuple[int, int, int]):
        """Calculate the output size after convolutions"""
        c, h, w = input_shape
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Apply conv layers: 8x4, 4x2, 3x1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        return convw * convh * 64
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, observations: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass to generate embeddings
        
        Args:
            observations: Input observations (batch_size, channels, height, width)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            embeddings: State embeddings (batch_size, embedding_dim)
        """
        # Ensure observations are in correct format and range
        if observations.dtype != torch.float32:
            observations = observations.float()
        
        # Normalize pixel values if they're in [0, 255] range
        if observations.max() > 1.0:
            observations = observations / 255.0
        
        # Extract CNN features
        cnn_features = self.cnn(observations)
        
        # Generate embeddings
        embeddings = self.embedding_head(cnn_features)
        
        # L2 normalization for stable similarity computation
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class SharedCNNEmbedding(nn.Module):
    """
    Shared CNN Embedding Network that reuses features from existing R2D2/DQN network
    
    This is more memory efficient as it doesn't duplicate the CNN layers.
    Can be used when you already have a trained R2D2 network and want to add
    embedding capability without retraining the CNN features.
    """
    
    def __init__(self, 
                 cnn_feature_dim: int = 512,
                 embedding_dim: int = 128,
                 use_batch_norm: bool = True):
        """
        Initialize shared embedding network
        
        Args:
            cnn_feature_dim: Input CNN feature dimension
            embedding_dim: Output embedding dimension
            use_batch_norm: Whether to use batch normalization
        """
        super(SharedCNNEmbedding, self).__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.embedding_dim = embedding_dim
        
        # Embedding head only (avoid BatchNorm for single batch compatibility)
        layers = []
        
        # First projection
        layers.append(nn.Linear(cnn_feature_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Second projection
        layers.append(nn.Linear(256, embedding_dim))
        
        self.embedding_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, cnn_features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass using pre-computed CNN features
        
        Args:
            cnn_features: CNN features (batch_size, cnn_feature_dim)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            embeddings: State embeddings (batch_size, embedding_dim)
        """
        # Generate embeddings
        embeddings = self.embedding_head(cnn_features)
        
        # L2 normalization
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training embedding networks
    
    Helps learn embeddings where similar states are close and 
    different states are far apart in embedding space.
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, 
                embedding2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embedding1: First set of embeddings (batch_size, embedding_dim)
            embedding2: Second set of embeddings (batch_size, embedding_dim) 
            labels: Binary labels (1 for similar, 0 for dissimilar)
            
        Returns:
            loss: Contrastive loss
        """
        # Compute euclidean distance
        distances = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_similar = labels * distances.pow(2)
        loss_dissimilar = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (loss_similar + loss_dissimilar).mean()
        return loss


def test_embedding_network():
    """Test function for embedding network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 4
    input_shape = (4, 84, 84)  # Atari frame stack
    embedding_dim = 128
    
    # Create network
    embedding_net = StateEmbeddingNetwork(
        input_shape=input_shape,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Test input
    observations = torch.randint(0, 256, (batch_size, *input_shape), 
                                dtype=torch.float32, device=device)
    
    # Forward pass
    with torch.no_grad():
        embeddings = embedding_net(observations)
    
    print(f"Input shape: {observations.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norms: {torch.norm(embeddings, p=2, dim=1)}")
    
    # Test shared CNN version
    shared_net = SharedCNNEmbedding(
        cnn_feature_dim=512,
        embedding_dim=embedding_dim
    ).to(device)
    
    # Simulate CNN features
    cnn_features = torch.randn(batch_size, 512, device=device)
    
    with torch.no_grad():
        shared_embeddings = shared_net(cnn_features)
    
    print(f"Shared embedding shape: {shared_embeddings.shape}")
    print(f"Shared embedding norms: {torch.norm(shared_embeddings, p=2, dim=1)}")
    
    print("✓ Embedding networks test passed!")


if __name__ == "__main__":
    test_embedding_network()