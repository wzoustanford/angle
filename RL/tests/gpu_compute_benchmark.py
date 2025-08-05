#!/usr/bin/env python3
"""
Quick GPU vs CPU computational benchmark for DQN components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from config.AgentConfig import AgentConfig
from model.dqn_network import DQN
from model.r2d2_network import R2D2Network
from model.device_utils import get_device_manager

def benchmark_network_inference(device_type: str, network, batch_size: int = 32, num_iterations: int = 100):
    """Benchmark network forward passes"""
    devmgr = get_device_manager(device_type)
    network = devmgr.to_dev(network)
    
    # Create dummy batch
    dummy_input = devmgr.to_dev(torch.randn(batch_size, 12, 210, 160))  # 4 stacked RGB frames
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = network(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device_type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = network(dummy_input)
    
    torch.cuda.synchronize() if device_type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_batch = total_time / num_iterations
    
    return time_per_batch * 1000  # Convert to milliseconds

def benchmark_training_step(device_type: str, network, batch_size: int = 32, num_iterations: int = 50):
    """Benchmark training forward+backward passes"""
    devmgr = get_device_manager(device_type)
    network = devmgr.to_dev(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    # Create dummy training batch
    states = devmgr.to_dev(torch.randn(batch_size, 12, 210, 160))
    actions = devmgr.to_dev(torch.randint(0, 6, (batch_size,)))
    targets = devmgr.to_dev(torch.randn(batch_size, 6))
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        q_values = network(states)
        loss = torch.nn.functional.mse_loss(q_values, targets)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    torch.cuda.synchronize() if device_type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        q_values = network(states)
        loss = torch.nn.functional.mse_loss(q_values, targets)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if device_type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / num_iterations
    
    return time_per_step * 1000  # Convert to milliseconds

def run_benchmark():
    """Run comprehensive benchmark"""
    print("DQN GPU vs CPU Computational Benchmark")
    print("=" * 50)
    
    # Test configurations
    obs_shape = (12, 210, 160)  # 4 stacked RGB frames
    n_actions = 6  # Space Invaders actions
    
    results = {}
    
    for device in ['cuda', 'cpu']:
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Skipping {device.upper()} - not available")
            continue
            
        print(f"\n=== {device.upper()} Benchmark ===")
        devmgr = get_device_manager(device)
        print(f"Device: {devmgr.device}")
        
        results[device] = {}
        
        # Test DQN network
        print("Testing DQN Network...")
        dqn = DQN(obs_shape, n_actions)
        
        inference_time = benchmark_network_inference(device, dqn, batch_size=32, num_iterations=100)
        training_time = benchmark_training_step(device, dqn, batch_size=32, num_iterations=50)
        
        results[device]['dqn_inference_ms'] = inference_time
        results[device]['dqn_training_ms'] = training_time
        
        print(f"  DQN Inference: {inference_time:.2f}ms per batch")
        print(f"  DQN Training: {training_time:.2f}ms per step")
        
        # Test R2D2 network
        print("Testing R2D2 Network...")
        r2d2 = R2D2Network(obs_shape, n_actions, lstm_size=256)  # Smaller LSTM for speed
        
        # For R2D2, we need sequence input
        devmgr_r2d2 = get_device_manager(device)
        r2d2 = devmgr_r2d2.to_dev(r2d2)
        
        # Benchmark R2D2 inference
        dummy_seq = devmgr_r2d2.to_dev(torch.randn(8, 10, 12, 210, 160))  # batch, seq, channels, h, w
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(20):  # Fewer iterations for R2D2
            with torch.no_grad():
                _, _ = r2d2.forward(dummy_seq)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        r2d2_time = (time.time() - start_time) / 20 * 1000
        
        results[device]['r2d2_inference_ms'] = r2d2_time
        print(f"  R2D2 Inference: {r2d2_time:.2f}ms per sequence batch")
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    if 'cuda' in results and 'cpu' in results:
        gpu_res = results['cuda']
        cpu_res = results['cpu']
        
        print(f"\nDQN Inference:")
        print(f"  GPU: {gpu_res['dqn_inference_ms']:.2f}ms")
        print(f"  CPU: {cpu_res['dqn_inference_ms']:.2f}ms")
        speedup = cpu_res['dqn_inference_ms'] / gpu_res['dqn_inference_ms']
        print(f"  Speedup: {speedup:.2f}x")
        
        print(f"\nDQN Training:")
        print(f"  GPU: {gpu_res['dqn_training_ms']:.2f}ms")
        print(f"  CPU: {cpu_res['dqn_training_ms']:.2f}ms")
        speedup = cpu_res['dqn_training_ms'] / gpu_res['dqn_training_ms']
        print(f"  Speedup: {speedup:.2f}x")
        
        print(f"\nR2D2 Inference:")
        print(f"  GPU: {gpu_res['r2d2_inference_ms']:.2f}ms")
        print(f"  CPU: {cpu_res['r2d2_inference_ms']:.2f}ms")
        speedup = cpu_res['r2d2_inference_ms'] / gpu_res['r2d2_inference_ms']
        print(f"  Speedup: {speedup:.2f}x")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\nPeak GPU memory: {memory_gb:.2f}GB")
    
    return results

if __name__ == "__main__":
    run_benchmark()