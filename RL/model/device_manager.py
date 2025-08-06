import torch
import os
import gc
import warnings
import psutil
from typing import Optional, Dict, Any, List
import logging


class DeviceManager:
    """
    Centralized device and memory management for all RL algorithms.
    
    Features:
    - Automatic GPU detection and selection
    - Memory management for 16GB GPU
    - Mixed precision training support
    - Memory monitoring and cleanup
    - Multi-GPU support preparation
    """
    
    def __init__(self, preferred_device: Optional[str] = None, enable_mixed_precision: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Device selection
        self.device = self._select_device(preferred_device)
        self.device_name = self._get_device_name()
        self.memory_info = self._get_memory_info()
        
        # Mixed precision setup
        self.mixed_precision = enable_mixed_precision and self.device.type == 'cuda'
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Memory management settings for 16GB GPU
        self._setup_memory_management()
        
        # Track allocated models for memory monitoring
        self.allocated_models: Dict[str, torch.nn.Module] = {}
        
        self._print_device_info()
    
    def _select_device(self, preferred_device: Optional[str]) -> torch.device:
        """Select the best available device"""
        if preferred_device:
            try:
                device = torch.device(preferred_device)
                if device.type == 'cuda' and torch.cuda.is_available():
                    return device
                elif device.type == 'cpu':
                    return device
                else:
                    self.logger.warning(f"Preferred device {preferred_device} not available, falling back to auto-selection")
            except Exception as e:
                self.logger.warning(f"Invalid preferred device {preferred_device}: {e}")
        
        # Auto-select best device
        if torch.cuda.is_available():
            # Select GPU with most free memory
            best_gpu = 0
            max_free_memory = 0
            
            for i in range(torch.cuda.device_count()):
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = i
            
            return torch.device(f'cuda:{best_gpu}')
        else:
            return torch.device('cpu')
    
    def _get_device_name(self) -> str:
        """Get human-readable device name"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(self.device)
        else:
            return "CPU"
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information"""
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            total_memory_gb = props.total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            free_memory_gb = total_memory_gb - allocated_memory_gb
            
            return {
                'total_gb': total_memory_gb,
                'allocated_gb': allocated_memory_gb,
                'free_gb': free_memory_gb
            }
        else:
            # CPU memory info
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'allocated_gb': (memory.total - memory.available) / (1024**3),
                'free_gb': memory.available / (1024**3)
            }
    
    def _setup_memory_management(self):
        """Setup memory management for 16GB GPU"""
        if self.device.type == 'cuda':
            # Set memory fraction to prevent OOM (use ~90% of available memory)
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            target_memory = int(total_memory * 0.9)
            
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for performance
            
            # Set environment variables for memory optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            self.logger.info(f"GPU memory management enabled for {self.memory_info['total_gb']:.1f}GB GPU")
    
    def _print_device_info(self):
        """Print device and memory information"""
        print(f"🔧 Device Manager Initialized")
        print(f"   Device: {self.device} ({self.device_name})")
        print(f"   Memory: {self.memory_info['free_gb']:.1f}GB free / {self.memory_info['total_gb']:.1f}GB total")
        if self.mixed_precision:
            print(f"   Mixed Precision: Enabled (AMP)")
        else:
            print(f"   Mixed Precision: Disabled")
    
    def to_device(self, tensor_or_model, model_name: Optional[str] = None):
        """Move tensor or model to managed device"""
        if isinstance(tensor_or_model, torch.nn.Module):
            # Move model to device
            model = tensor_or_model.to(self.device)
            
            # Track model for memory monitoring
            if model_name:
                self.allocated_models[model_name] = model
            
            # Enable mixed precision if supported
            if self.mixed_precision and hasattr(model, 'half'):
                # Only use half precision for inference, keep training in mixed precision
                pass  # AMP handles this automatically during forward pass
            
            return model
        else:
            # Move tensor to device
            return tensor_or_model.to(self.device)
    
    def get_optimal_batch_size(self, model_name: str, base_batch_size: int = 32, 
                              input_shape: tuple = (4, 84, 84)) -> int:
        """Calculate optimal batch size based on available memory"""
        if self.device.type == 'cpu':
            return base_batch_size
        
        available_memory_gb = self.memory_info['free_gb']
        
        # Estimate memory usage per sample (rough heuristic)
        # Input: ~4 * 84 * 84 * 4 bytes = ~113KB per frame stack
        # Network parameters and gradients: varies by model
        # Activations: depends on network depth
        
        memory_per_sample_mb = 1.0  # Conservative estimate: 1MB per sample
        
        if 'r2d2' in model_name.lower() or 'lstm' in model_name.lower():
            memory_per_sample_mb *= 2.0  # LSTM requires more memory
        
        if 'distributed' in model_name.lower():
            memory_per_sample_mb *= 1.5  # Distributed training overhead
        
        # Calculate optimal batch size (use 70% of available memory)
        max_batch_size = int((available_memory_gb * 1024 * 0.7) / memory_per_sample_mb)
        optimal_batch_size = min(max_batch_size, base_batch_size * 2)  # Don't exceed 2x base
        optimal_batch_size = max(optimal_batch_size, 8)  # Minimum batch size of 8
        
        # Round to nearest power of 2 for efficiency
        optimal_batch_size = 2 ** int(torch.log2(torch.tensor(optimal_batch_size)))
        
        if optimal_batch_size != base_batch_size:
            self.logger.info(f"Adjusted batch size for {model_name}: {base_batch_size} → {optimal_batch_size}")
        
        return optimal_batch_size
    
    def optimize_model(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Apply model optimizations for GPU training"""
        model = self.to_device(model, model_name)
        
        if self.device.type == 'cuda':
            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                try:
                    model = torch.compile(model, mode='default')
                    self.logger.info(f"Model {model_name} compiled for optimization")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed for {model_name}: {e}")
        
        return model
    
    def create_optimizer(self, model: torch.nn.Module, learning_rate: float = 1e-4, 
                        optimizer_type: str = 'adam') -> torch.optim.Optimizer:
        """Create optimized optimizer for the device"""
        if optimizer_type.lower() == 'adam':
            # Use fused Adam for better GPU performance
            if self.device.type == 'cuda':
                try:
                    optimizer = torch.optim.Adam(
                        model.parameters(), 
                        lr=learning_rate,
                        fused=True  # Fused operations for better GPU performance
                    )
                    self.logger.info("Using fused Adam optimizer for GPU")
                    return optimizer
                except Exception:
                    pass  # Fall back to regular Adam
            
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        elif optimizer_type.lower() == 'adamw':
            if self.device.type == 'cuda':
                try:
                    return torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rate,
                        fused=True
                    )
                except Exception:
                    pass
            return torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def forward_with_amp(self, model: torch.nn.Module, *args, **kwargs):
        """Forward pass with automatic mixed precision"""
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                return model(*args, **kwargs)
        else:
            return model(*args, **kwargs)
    
    def backward_with_amp(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with automatic mixed precision"""
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': total - reserved,
                'total_gb': total,
                'utilization': (reserved / total) * 100
            }
        else:
            memory = psutil.virtual_memory()
            return {
                'allocated_gb': (memory.total - memory.available) / (1024**3),
                'free_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'utilization': memory.percent
            }
    
    def print_memory_usage(self):
        """Print current memory usage"""
        usage = self.get_memory_usage()
        print(f"💾 Memory Usage: {usage['allocated_gb']:.1f}GB allocated, "
              f"{usage['free_gb']:.1f}GB free ({usage['utilization']:.1f}% used)")
    
    def monitor_memory(self, threshold_percent: float = 85.0) -> bool:
        """Monitor memory usage and warn if threshold exceeded"""
        usage = self.get_memory_usage()
        
        if usage['utilization'] > threshold_percent:
            self.logger.warning(f"High memory usage: {usage['utilization']:.1f}% "
                              f"(threshold: {threshold_percent}%)")
            return True
        return False
    
    def suggest_memory_optimization(self) -> List[str]:
        """Suggest memory optimization strategies"""
        suggestions = []
        usage = self.get_memory_usage()
        
        if usage['utilization'] > 80:
            suggestions.append("Reduce batch size")
            suggestions.append("Enable gradient checkpointing")
            suggestions.append("Use mixed precision training")
            
        if usage['utilization'] > 90:
            suggestions.append("Clear cache between episodes")
            suggestions.append("Consider using CPU for some operations")
            
        if len(self.allocated_models) > 3:
            suggestions.append("Consider model sharing between agents")
            
        return suggestions


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(preferred_device: Optional[str] = None, 
                      enable_mixed_precision: bool = True) -> DeviceManager:
    """Get or create global device manager instance"""
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(preferred_device, enable_mixed_precision)
    
    return _global_device_manager


def reset_device_manager():
    """Reset global device manager (useful for testing)"""
    global _global_device_manager
    if _global_device_manager is not None:
        _global_device_manager.clear_cache()
    _global_device_manager = None