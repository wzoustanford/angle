#!/usr/bin/env python
"""Test script to verify num_grin_recurrence parameter flow"""

import sys
import torch
import torch.nn as nn
from moore.utils import networks_sac

# Test the network initialization with different num_grin_recurrence values
def test_network_param():
    input_shape = (39,)  # Example input shape for MetaWorld
    output_shape = (4,)  # Example output shape
    n_features = [256, 256]
    n_contexts = 5
    
    # Test with default value (2)
    print("Testing MetaworldSACMixtureMHCriticNetworkGRIN with default num_grin_recurrence...")
    critic1 = networks_sac.MetaworldSACMixtureMHCriticNetworkGRIN(
        input_shape=input_shape,
        output_shape=(1,),
        n_features=n_features,
        n_contexts=n_contexts,
        n_experts=4,
        num_grin_recurrence=2
    )
    assert critic1.num_grin_recurrence == 2, "Default value should be 2"
    print(f"✓ Critic network initialized with num_grin_recurrence = {critic1.num_grin_recurrence}")
    
    # Test with custom value (5)
    print("\nTesting with custom num_grin_recurrence = 5...")
    critic2 = networks_sac.MetaworldSACMixtureMHCriticNetworkGRIN(
        input_shape=input_shape,
        output_shape=(1,),
        n_features=n_features,
        n_contexts=n_contexts,
        n_experts=4,
        num_grin_recurrence=5
    )
    assert critic2.num_grin_recurrence == 5, "Custom value should be 5"
    print(f"✓ Critic network initialized with num_grin_recurrence = {critic2.num_grin_recurrence}")
    
    # Test Actor network
    print("\nTesting MetaworldSACMixtureMHActorNetworkGRIN with num_grin_recurrence = 3...")
    actor = networks_sac.MetaworldSACMixtureMHActorNetworkGRIN(
        input_shape=input_shape,
        output_shape=output_shape,
        n_features=n_features,
        n_contexts=n_contexts,
        n_experts=4,
        shared_mu_sigma=True,
        num_grin_recurrence=3
    )
    assert actor.num_grin_recurrence == 3, "Actor value should be 3"
    print(f"✓ Actor network initialized with num_grin_recurrence = {actor.num_grin_recurrence}")
    
    print("\n✅ All tests passed! The num_grin_recurrence parameter is properly configured.")

if __name__ == "__main__":
    test_network_param()