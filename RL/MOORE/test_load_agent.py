#!/usr/bin/env python3
"""
Test script to debug agent loading
"""

import os
import sys
import pickle

# Add the MOORE directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting agent load test...")

# Test basic imports
print("Importing modules...")
import metaworld
from moore.algorithms.actor_critic import MTSAC
print("Imports successful")

# Load args
checkpoint_dir = "logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_3e_pretexTrue/seed_42"
args_path = os.path.join(os.path.dirname(checkpoint_dir), 'args.pkl')

print(f"Loading args from: {args_path}")
with open(args_path, 'rb') as f:
    args = pickle.load(f)
print(f"Args loaded. n_experts={args.n_experts}")

# Try to load the agent
agent_path = os.path.join(checkpoint_dir, 'agent', 'agent_6')
print(f"Loading agent from: {agent_path}")

try:
    agent = MTSAC.load(agent_path)
    print("Agent loaded successfully!")
    
    # Check agent structure
    print(f"Agent type: {type(agent)}")
    print(f"Has policy: {hasattr(agent, 'policy')}")
    if hasattr(agent, 'policy'):
        print(f"Policy type: {type(agent.policy)}")
    
    # Try to set missing attributes
    if hasattr(agent.policy._approximator.model.network, '_h'):
        print("Setting get_activation_list...")
        agent.policy._approximator.model.network.get_activation_list = [
            f'1.model_layers.{str(i)}.act_0' for i in range(args.n_experts)
        ]
        
        if not hasattr(agent.policy._approximator.model.network, 'use_pretex_inhibition'):
            print("Setting use_pretex_inhibition...")
            has_pretex = hasattr(agent.policy._approximator.model.network, 'pretex_inhibition_network')
            agent.policy._approximator.model.network.use_pretex_inhibition = has_pretex
            print(f"use_pretex_inhibition set to: {has_pretex}")
    
    print("Agent setup complete!")
    
except Exception as e:
    print(f"Error loading agent: {e}")
    import traceback
    traceback.print_exc()

print("Test complete")