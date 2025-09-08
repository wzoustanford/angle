#!/usr/bin/env python3
"""
Parse MT3 experiment logs to extract metrics from final epoch (epoch 10).
Extracts success rate, meanJ, and discounted meanJ for EACH TASK across 5 random seeds.
"""

import os
import re
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse a single log file to extract metrics from epoch 10."""
    
    metrics = {
        'success_rate': {},
        'mean_J': {},
        'mean_discounted_J': {}
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match epoch lines
    pattern = r'Epoch (\d+) \| C: (\S+) .* mean_J: ([\d.]+) mean_discounted_J: ([\d.]+) success_rate: ([\d.]+)'
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            task = match.group(2)
            mean_j = float(match.group(3))
            discounted_mean_j = float(match.group(4))
            success_rate = float(match.group(5))
            
            # Only keep epoch 10 data
            if epoch == 10:
                metrics['success_rate'][task] = success_rate
                metrics['mean_J'][task] = mean_j
                metrics['mean_discounted_J'][task] = discounted_mean_j
    
    return metrics

def main():
    # Base directory
    base_dir = Path("/home/ubuntu/code/angle/RL/MOORE/logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e_pretexFalse")
    
    # Seeds to process
    seeds = [42, 123, 456, 789, 1011]
    
    # Tasks we expect to find
    tasks = ['reach-v2', 'push-v2', 'pick-place-v2']
    
    # Store results for each task separately
    task_results = {task: {
        'success_rate': [],
        'mean_J': [],
        'mean_discounted_J': []
    } for task in tasks}
    
    print("Parsing logs from MT3 experiment...")
    print("=" * 60)
    
    # Process each seed
    for seed in seeds:
        log_file = base_dir / f"seed_{seed}" / f"seed_{seed}.log"
        
        if not log_file.exists():
            print(f"Warning: Log file not found for seed {seed}: {log_file}")
            continue
        
        print(f"\nProcessing seed {seed}...")
        metrics = parse_log_file(log_file)
        
        # Store metrics for each task
        for task in tasks:
            if task in metrics['success_rate']:
                task_results[task]['success_rate'].append(metrics['success_rate'][task])
                task_results[task]['mean_J'].append(metrics['mean_J'][task])
                task_results[task]['mean_discounted_J'].append(metrics['mean_discounted_J'][task])
                
                print(f"  {task}:")
                print(f"    Success Rate: {metrics['success_rate'][task]:.4f}")
                print(f"    Mean J: {metrics['mean_J'][task]:.4f}")
                print(f"    Discounted Mean J: {metrics['mean_discounted_J'][task]:.4f}")
    
    # Calculate statistics for each task across seeds
    print("\n" + "=" * 60)
    print("FINAL STATISTICS (Epoch 10, per task across 5 seeds):")
    print("=" * 60)
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        print("-" * 40)
        
        if task_results[task]['success_rate']:
            success_rates = task_results[task]['success_rate']
            mean_js = task_results[task]['mean_J']
            disc_mean_js = task_results[task]['mean_discounted_J']
            
            print(f"1. Success Rate:")
            print(f"   Mean ± Std: {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f}")
            print(f"   Values: {[f'{v:.4f}' for v in success_rates]}")
            
            print(f"\n2. Mean J:")
            print(f"   Mean ± Std: {np.mean(mean_js):.4f} ± {np.std(mean_js):.4f}")
            print(f"   Values: {[f'{v:.4f}' for v in mean_js]}")
            
            print(f"\n3. Discounted Mean J:")
            print(f"   Mean ± Std: {np.mean(disc_mean_js):.4f} ± {np.std(disc_mean_js):.4f}")
            print(f"   Values: {[f'{v:.4f}' for v in disc_mean_js]}")
        else:
            print("   No data found for this task!")

if __name__ == "__main__":
    main()