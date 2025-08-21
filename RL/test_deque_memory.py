#!/usr/bin/env python3
"""Test if deque is causing memory issues"""

from collections import deque
import numpy as np
import psutil
import gc

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing deque memory usage")
print("="*50)

# Create states similar to Atari
def create_state():
    return np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)

initial_mem = get_mem_mb()
print(f"Initial memory: {initial_mem:.1f}MB")

# Test 1: Simple list
states_list = []
for i in range(1000):
    states_list.append(create_state())
    
mem_after_list = get_mem_mb()
print(f"After 1000 states in list: {mem_after_list:.1f}MB (+{mem_after_list-initial_mem:.1f}MB)")

del states_list
gc.collect()

# Test 2: Deque with maxlen
states_deque = deque(maxlen=1000)
mem_before_deque = get_mem_mb()

for i in range(5000):  # Add 5000 but keep only last 1000
    states_deque.append(create_state())
    if (i+1) % 1000 == 0:
        current_mem = get_mem_mb()
        print(f"After {i+1} additions to deque: {current_mem:.1f}MB (+{current_mem-mem_before_deque:.1f}MB)")
        gc.collect()  # Force garbage collection

final_mem = get_mem_mb()
print(f"\nFinal memory: {final_mem:.1f}MB")
print(f"Deque should only have 1000 items but memory increased by {final_mem-mem_before_deque:.1f}MB")

# Check actual size
actual_size = sum(state.nbytes for state in states_deque) / 1024 / 1024
print(f"Actual data size in deque: {actual_size:.1f}MB")
print("="*50)