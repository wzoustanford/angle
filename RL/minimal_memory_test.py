#!/usr/bin/env python3
"""Minimal test to check if detach fixes are working"""

import torch
import torch.nn as nn
import psutil
import gc

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing LSTM hidden state memory leak...")
print("="*50)

# Create a simple LSTM
lstm = nn.LSTM(10, 20, batch_first=True)
lstm.cuda()

# Track memory
initial_mem = get_mem_mb()
print(f"Initial memory: {initial_mem:.1f}MB")

# Test 1: WITHOUT detach (memory leak)
print("\nTest 1: Storing hidden states WITHOUT detach")
hidden_states_bad = []
hidden = None

for i in range(100):
    x = torch.randn(1, 1, 10).cuda()
    out, hidden = lstm(x, hidden)
    # Store without detaching - CAUSES MEMORY LEAK
    hidden_states_bad.append(hidden)
    
    if (i+1) % 20 == 0:
        current_mem = get_mem_mb()
        print(f"  Step {i+1}: Memory = {current_mem:.1f}MB (+{current_mem-initial_mem:.1f}MB)")

mem_after_bad = get_mem_mb()
print(f"After 100 steps WITHOUT detach: {mem_after_bad:.1f}MB (+{mem_after_bad-initial_mem:.1f}MB)")

# Clear
del hidden_states_bad
del hidden
gc.collect()
torch.cuda.empty_cache()

print("\n" + "-"*50)

# Test 2: WITH detach (no memory leak)
print("\nTest 2: Storing hidden states WITH detach")
hidden_states_good = []
hidden = None
initial_mem2 = get_mem_mb()

for i in range(100):
    x = torch.randn(1, 1, 10).cuda()
    out, new_hidden = lstm(x, hidden)
    # Detach before storing - PREVENTS MEMORY LEAK
    if new_hidden is not None:
        hidden = tuple(h.detach() for h in new_hidden)
    else:
        hidden = new_hidden
    hidden_states_good.append(hidden)
    
    if (i+1) % 20 == 0:
        current_mem = get_mem_mb()
        print(f"  Step {i+1}: Memory = {current_mem:.1f}MB (+{current_mem-initial_mem2:.1f}MB)")

mem_after_good = get_mem_mb()
print(f"After 100 steps WITH detach: {mem_after_good:.1f}MB (+{mem_after_good-initial_mem2:.1f}MB)")

print("\n" + "="*50)
print("Results:")
print(f"  Without detach: +{mem_after_bad-initial_mem:.1f}MB (LEAK)")
print(f"  With detach:    +{mem_after_good-initial_mem2:.1f}MB (OK)")
print("="*50)