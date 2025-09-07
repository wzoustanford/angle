#!/usr/bin/env python3
"""
Simple evaluation script for trained MetaWorld MT3 models with MOORE
This version recreates the agent from scratch with the correct architecture
"""

import os
import sys
import numpy as np
import pickle
import argparse
import torch
import torch.optim as optim

# Add the MOORE directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metaworld
from moore.core import VecCore
from moore.algorithms.actor_critic import MTSAC
from moore.environments.metaworld_env import make_env
from moore.environments import SubprocVecEnv
from moore.utils.dataset import get_stats
import moore.utils.networks_sac as Network


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
    
    # Recreate the agent with the same parameters as during training
    # Actor parameters
    actor_params = None
    actor_mu_params = None
    actor_sigma_params = None
    
    if args.shared_mu_sigma:
        actor_params = dict(
            network=getattr(Network, args.actor_network),
            n_features=args.actor_n_features,
            use_cuda=args.use_cuda,
            input_shape=mdp.info.observation_space.shape,
            output_shape=mdp.info.action_space.shape,
            n_experts=args.n_experts,
            orthogonal=args.orthogonal,
            activation=args.activation,
            agg_activation=args.agg_activation,
            n_contexts=n_contexts,
            shared_mu_sigma=args.shared_mu_sigma
        )
    else:
        actor_mu_params = dict(
            network=getattr(Network, args.actor_network),
            n_features=args.actor_n_features,
            use_cuda=args.use_cuda,
            input_shape=mdp.info.observation_space.shape,
            output_shape=mdp.info.action_space.shape,
            n_experts=args.n_experts,
            orthogonal=args.orthogonal,
            activation=args.activation,
            agg_activation=args.agg_activation,
            n_contexts=n_contexts
        )
        # For sigma network (usually same as mu)
        actor_sigma_params = dict(
            network=getattr(Network, args.actor_network),
            n_features=args.actor_n_features,
            use_cuda=args.use_cuda,
            input_shape=mdp.info.observation_space.shape,
            output_shape=mdp.info.action_space.shape,
            n_experts=args.n_experts,
            orthogonal=args.orthogonal,
            activation=args.activation,
            agg_activation=args.agg_activation,
            n_contexts=n_contexts
        )
    
    actor_optimizer = {
        'class': optim.Adam,
        'params': {'lr': args.lr_actor}
    }
    
    # Critic parameters
    critic_input_shape = (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],)
    
    critic_params = dict(
        network=getattr(Network, args.critic_network),
        optimizer={'class': optim.Adam, 'params': {'lr': args.lr_critic}},
        loss=torch.nn.functional.mse_loss,
        n_features=args.critic_n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
        use_cuda=args.use_cuda,
        n_experts=args.n_experts,
        orthogonal=args.orthogonal,
        activation=args.activation,
        agg_activation=args.agg_activation,
        n_contexts=n_contexts
    )
    
    # Create the agent
    agent = MTSAC(
        mdp.info, 
        actor_optimizer, 
        critic_params,
        batch_size=args.batch_size,
        initial_replay_size=args.initial_replay_size,
        max_replay_size=args.max_replay_size,
        warmup_transitions=args.warmup_transitions,
        tau=args.tau,
        lr_alpha=args.lr_alpha,
        actor_params=actor_params,
        actor_mu_params=actor_mu_params,
        actor_sigma_params=actor_sigma_params,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        shared_mu_sigma=args.shared_mu_sigma,
        n_contexts=n_contexts
    )
    
    # Load the saved weights
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
        print(f"Loading weights from: {agent_path}")
    
    # Load the saved agent weights
    loaded_agent = MTSAC.load(agent_path)
    
    # Transfer the learned parameters
    agent.policy._approximator.model.network.load_state_dict(
        loaded_agent.policy._approximator.model.network.state_dict()
    )
    agent._critic_approximator.model.network.load_state_dict(
        loaded_agent._critic_approximator.model.network.state_dict()
    )
    agent._target_critic_approximator.model.network.load_state_dict(
        loaded_agent._target_critic_approximator.model.network.state_dict()
    )
    
    # Transfer log_alpha values
    for i in range(n_contexts):
        agent.set_log_alpha(loaded_agent.get_log_alpha(i), i)
    
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
    
    # Calculate overall statistics
    overall_success_rate = np.mean([results[task]['success_rate'] for task in results])
    overall_mean_return = np.mean([results[task]['mean_return'] for task in results])
    
    if verbose:
        print("\n" + "="*50)
        print("OVERALL RESULTS:")
        print("="*50)
        print(f"Average Success Rate: {overall_success_rate*100:.1f}%")
        print(f"Average Return: {overall_mean_return:.2f}")
        print("\nPer-Task Results:")
        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
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