# main.py
import argparse
import torch
import numpy as np
import random
import os

from config import DQNConfig, PPOConfig, A3CConfig, RainbowConfig
from dqn import DQNAgent
from ppo import PPO
from a3c import train_a3c
from rainbow import RainbowAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents on Atari games')
    parser.add_argument('--algorithm', type=str, default='dqn', 
                       choices=['dqn', 'ppo', 'a3c', 'rainbow'],
                       help='Algorithm to use')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                       help='Atari environment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-steps', type=int, default=10_000_000,
                       help='Number of environment steps')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate a trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Algorithm-specific arguments
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=None,
                       help='Replay buffer size')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of parallel processes')
    
    return parser.parse_args()

def create_config(args):
    """Create configuration based on algorithm and arguments"""
    if args.algorithm == 'dqn':
        config = DQNConfig()
    elif args.algorithm == 'ppo':
        config = PPOConfig()
    elif args.algorithm == 'a3c':
        config = A3CConfig()
    elif args.algorithm == 'rainbow':
        config = RainbowConfig()
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Override config with command line arguments
    config.env_name = args.env
    config.seed = args.seed
    config.num_env_steps = args.num_steps
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None and hasattr(config, 'batch_size'):
        config.batch_size = args.batch_size
    if args.buffer_size is not None and hasattr(config, 'buffer_size'):
        config.buffer_size = args.buffer_size
    if args.num_processes is not None and hasattr(config, 'num_processes'):
        config.num_processes = args.num_processes
    
    return config

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train(args):
    """Train an agent"""
    config = create_config(args)
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set seeds
    set_seeds(config.seed)
    
    print(f"Training {args.algorithm} on {args.env}")
    print(f"Device: {config.device}")
    print(f"Number of steps: {config.num_env_steps:,}")
    
    # Create and train agent
    if args.algorithm == 'dqn':
        agent = DQNAgent(config)
        agent.train()
    elif args.algorithm == 'ppo':
        agent = PPO(config)
        agent.train()
    elif args.algorithm == 'a3c':
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
        train_a3c(config)
    elif args.algorithm == 'rainbow':
        agent = RainbowAgent(config)
        agent.train()
    
    # Final evaluation
    if args.algorithm != 'a3c':  # A3C does evaluation during training
        print("\nRunning final evaluation...")
        mean_reward, std_reward = agent.evaluate(num_episodes=30)
        print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")

def evaluate(args):
    """Evaluate a trained agent"""
    if args.checkpoint is None:
        raise ValueError("Must provide --checkpoint for evaluation")
    
    config = create_config(args)
    set_seeds(config.seed)
    
    print(f"Evaluating {args.algorithm} on {args.env}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create agent
    if args.algorithm == 'dqn':
        agent = DQNAgent(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
    elif args.algorithm == 'ppo':
        agent = PPO(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    elif args.algorithm == 'a3c':
        from networks import ActorCritic
        from envs import make_atari_env
        env = make_atari_env(config.env_name, config.seed, config.frame_stack)
        model = ActorCritic(config.frame_stack, env.action_space.n).to(config.device)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Create a simple evaluation function for A3C
        def evaluate_a3c(model, env, num_episodes=30):
            rewards = []
            for _ in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    # Handle LazyFrames
                    if hasattr(state, '_force'):
                        state_array = np.array(state._force())
                    else:
                        state_array = state
                    state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(config.device)
                    with torch.no_grad():
                        logits, _ = model(state_tensor)
                        action = logits.argmax(1).item()
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                rewards.append(episode_reward)
            return np.mean(rewards), np.std(rewards)
        mean_reward, std_reward = evaluate_a3c(model, env)
        env.close()
    elif args.algorithm == 'rainbow':
        agent = RainbowAgent(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    if args.algorithm != 'a3c':
        mean_reward, std_reward = agent.evaluate(num_episodes=30)
    
    print(f"\nEvaluation results over 30 episodes:")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

def main():
    args = parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
