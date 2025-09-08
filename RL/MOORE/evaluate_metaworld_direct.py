#!/usr/bin/env python3
"""
Direct evaluation script for trained MetaWorld MT3 models
This version directly loads the saved agent without recreating it
"""

import os
import sys
import numpy as np
import pickle
import argparse

# Add the MOORE directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metaworld
from moore.core import VecCore
from moore.algorithms.actor_critic import MTSAC
from moore.environments.metaworld_env import make_env
from moore.environments import SubprocVecEnv
from moore.utils.dataset import get_stats


def evaluate_model(checkpoint_dir, n_episodes=10, render=False, verbose=True):
    """
    Evaluate a trained MOORE model on MetaWorld MT3 tasks
    """
    
    # Load the saved arguments
    args_path = os.path.join(os.path.dirname(checkpoint_dir), 'args.pkl')
    if not os.path.exists(args_path):
        args_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'args.pkl')
    
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    
    # Setup the benchmark
    exp_type = args.exp_type  # Should be 'MT3'
    benchmark = getattr(metaworld, exp_type)()
    
    # Create environment
    env_names = list(benchmark.train_classes.keys())
    n_contexts = len(env_names)
    
    if verbose:
        print(f"\nEvaluating on {exp_type} with {n_contexts} tasks:")
        for i, name in enumerate(env_names):
            print(f"  Task {i+1}: {name}")
        print()
    
    # Setup MDP parameters
    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval if hasattr(args, 'gamma_eval') else 1.0
    
    # Create the vectorized environment
    mdp = SubprocVecEnv(
        [make_env(env_name=env_name, 
                  env_cls=env_cls, 
                  train_tasks=benchmark.train_tasks, 
                  horizon=horizon, 
                  gamma=gamma, 
                  normalize_reward=args.normalize_reward if hasattr(args, 'normalize_reward') else False,
                  sample_task_per_episode=args.sample_task_per_episode if hasattr(args, 'sample_task_per_episode') else False)
          for env_name, env_cls in benchmark.train_classes.items()])
    
    # Load the agent directly
    agent_path = os.path.join(checkpoint_dir, 'agent', 'agent_final')
    
    if not os.path.exists(agent_path):
        # Try to find any agent checkpoint
        agent_dir = os.path.join(checkpoint_dir, 'agent')
        if os.path.exists(agent_dir):
            available_agents = [f for f in os.listdir(agent_dir) if f.startswith('agent_')]
            if available_agents:
                available_agents.sort()
                agent_path = os.path.join(agent_dir, available_agents[-1])
                print(f"Using checkpoint: {agent_path}")
    
    if verbose:
        print(f"Loading agent from: {agent_path}")
    
    # Load the saved agent directly
    agent = MTSAC.load(agent_path)
    
    # IMPORTANT: Set the missing attribute that the saved model needs
    if hasattr(agent.policy._approximator.model.network, '_h'):
        # The network needs get_activation_list which isn't saved
        agent.policy._approximator.model.network.get_activation_list = [
            f'1.model_layers.{str(i)}.act_0' for i in range(args.n_experts)
        ]
    
    # Create the core
    core = VecCore(agent, mdp)
    
    # Evaluate each task
    results = {}
    core.eval = True
    
    for task_idx, task_name in enumerate(env_names):
        if verbose:
            print(f"\nEvaluating Task {task_idx+1}/{n_contexts}: {task_name}")
        
        core.current_idx = task_idx
        
        # Run evaluation episodes
        try:
            dataset, dataset_info = core.evaluate(
                n_episodes=n_episodes, 
                render=render, 
                get_env_info=True
            )
            
            # Calculate statistics
            min_J, max_J, mean_J, mean_discounted_J, success_rate = get_stats(
                dataset, gamma, gamma_eval, dataset_info=dataset_info
            )
            
            results[task_name] = {
                'min_return': min_J,
                'max_return': max_J,
                'mean_return': mean_J,
                'mean_discounted_return': mean_discounted_J,
                'success_rate': success_rate
            }
            
            if verbose:
                print(f"  Mean Return: {mean_J:.2f}")
                print(f"  Success Rate: {success_rate*100:.1f}%")
        except Exception as e:
            print(f"  Error evaluating {task_name}: {e}")
            results[task_name] = {
                'min_return': 0,
                'max_return': 0,
                'mean_return': 0,
                'mean_discounted_return': 0,
                'success_rate': 0,
                'error': str(e)
            }
    
    # Calculate overall statistics
    valid_results = [r for r in results.values() if 'error' not in r]
    if valid_results:
        overall_success_rate = np.mean([r['success_rate'] for r in valid_results])
        overall_mean_return = np.mean([r['mean_return'] for r in valid_results])
    else:
        overall_success_rate = 0
        overall_mean_return = 0
    
    if verbose:
        print("\n" + "="*50)
        print("OVERALL RESULTS:")
        print("="*50)
        print(f"Average Success Rate: {overall_success_rate*100:.1f}%")
        print(f"Average Return: {overall_mean_return:.2f}")
        print("\nPer-Task Results:")
        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            if 'error' in task_results:
                print(f"  Error: {task_results['error']}")
            else:
                print(f"  Return: {task_results['mean_return']:.2f} (min: {task_results['min_return']:.2f}, max: {task_results['max_return']:.2f})")
                print(f"  Success Rate: {task_results['success_rate']*100:.1f}%")
    
    # Close the environment
    mdp.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MOORE models on MetaWorld MT3')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e/seed_0',
                        help='Path to the checkpoint directory')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of episodes to evaluate per task')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Make path absolute if it's relative
    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join(os.getcwd(), args.checkpoint_dir)
    
    print(f"Evaluating model from: {args.checkpoint_dir}")
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_dir=args.checkpoint_dir,
        n_episodes=args.n_episodes,
        render=args.render,
        verbose=not args.quiet
    )
    
    # Save results to file
    results_file = os.path.join(args.checkpoint_dir, f'evaluation_results_{args.n_episodes}ep.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()