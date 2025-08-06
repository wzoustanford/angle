import torch
from typing import Optional


class DeviceManager:
    """Simple device management - just handles CPU vs GPU placement"""
    
    def __init__(self, preferred_device: Optional[str] = None):
        # Select device
        if preferred_device:
            self.device = torch.device(preferred_device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"🔧 Device: {self.device}")
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU: {torch.cuda.get_device_name(self.device)} ({memory_gb:.1f}GB)")
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to device"""
        return tensor_or_model.to(self.device)
    
    def to_dev(self, tensor_or_model):
        """Short alias for to_device"""
        return self.to_device(tensor_or_model)


# Global instance
_device_manager = None

def get_device_manager(preferred_device: Optional[str] = None) -> DeviceManager:
    """Get global device manager"""
    global _device_manager
    if _device_manager is None or preferred_device is not None:
        _device_manager = DeviceManager(preferred_device)
    return _device_manager


def to_device(tensor_or_model, device_manager: Optional[DeviceManager] = None):
    """Convenience function to move tensor/model to device"""
    if device_manager is None:
        device_manager = get_device_manager()
    return device_manager.to_device(tensor_or_model)