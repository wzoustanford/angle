import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ale_py
from typing import Dict, Optional, Tuple
import os
from PIL import Image

from .muzero_network import MuZeroNetwork, scalar_to_support, support_to_scalar
from .muzero_mcts import MCTS
from .muzero_buffer import MuZeroReplayBuffer, GameHistory
from .device_utils import get_device_manager

# Register Atari environments
gym.register_envs(ale_py)


class MuZeroAgent:
    """MuZero Agent for Atari games"""
    
    def __init__(self, config):
        self.config = config
        self.devmgr = get_device_manager(getattr(config, 'device', None))
        self.device = self.devmgr.device
        
        # Environment setup
        self.env = gym.make(config.env_name)
        self.action_space_size = self.env.action_space.n
        config.action_space_size = self.action_space_size  # Update config
        
        # Networks
        self.network = self.devmgr.to_dev(MuZeroNetwork(config))
        self.target_network = self.devmgr.to_dev(MuZeroNetwork(config))
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # MCTS
        self.mcts = MCTS(config)
        
        # Replay buffer
        self.replay_buffer = MuZeroReplayBuffer(config)
        
        # Training statistics
        self.training_steps = 0
        self.games_played = 0
        
        # Temperature schedule - create a proper function
        def temperature_fn(training_steps):
            if training_steps < 50000:
                return 1.0
            elif training_steps < 75000:
                return 0.5
            else:
                return 0.25
        self.visit_softmax_temperature_fn = temperature_fn
        
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocess raw observation from environment"""
        # Resize observation to configured size
        if len(observation.shape) == 3:
            # Convert to PIL Image for resizing
            img = Image.fromarray(observation)
            img = img.resize((self.config.observation_shape[2], 
                            self.config.observation_shape[1]))
            observation = np.array(img)
            
            # Transpose to channels-first and normalize
            observation = observation.transpose(2, 0, 1) / 255.0
        
        return torch.FloatTensor(observation)
    
    def self_play(self, render: bool = False) -> GameHistory:
        """
        Play a self-play game and store in replay buffer
        
        Args:
            render: Whether to render the game
            
        Returns:
            GameHistory object with game data
        """
        game = GameHistory()
        observation, _ = self.env.reset()
        done = False
        
        # Temperature for exploration
        temperature = self.visit_softmax_temperature_fn(self.training_steps)
        
        print(f"  Starting self-play game {self.games_played + 1}, max_moves={self.config.max_moves}")
        step_count = 0
        
        # For Breakout, we need to FIRE at the beginning to start the game
        if 'Breakout' in self.config.env_name:
            # Always start with FIRE action in Breakout
            print(f"    Taking initial FIRE action to start Breakout")
            for _ in range(3):  # Take FIRE action a few times to ensure game starts
                observation, reward, terminated, truncated, _ = self.env.step(1)  # Action 1 = FIRE
                if terminated or truncated:
                    observation, _ = self.env.reset()
        
        # Track action distribution for debugging
        action_counts = {i: 0 for i in range(self.action_space_size)}
        
        while not done and len(game) < self.config.max_moves:
            # Log progress every 100 steps
            if step_count % 100 == 0 and step_count > 0:
                print(f"    Step {step_count}/{self.config.max_moves}, reward so far: {sum(game.rewards):.1f}")
            step_count += 1
            # Preprocess observation
            obs_tensor = self.preprocess_observation(observation).to(self.device)
            
            # SIMPLIFIED: No MCTS, just use network prediction with high exploration
            # 50% random actions for exploration (increased from 25%)
            if np.random.random() < 0.5:
                # Random action with emphasis on non-NOOP actions
                if np.random.random() < 0.7:  # 70% chance to pick non-NOOP
                    action = np.random.choice([1, 2, 3])  # FIRE, RIGHT, LEFT
                else:
                    action = 0  # NOOP
                
                action_probs = np.zeros(self.action_space_size)
                action_probs[action] = 1.0
                value = 0.0
            else:
                # Use network prediction directly (no MCTS)
                with torch.no_grad():
                    output = self.network.initial_inference(obs_tensor.unsqueeze(0))
                    policy_logits = output['policy_logits'].squeeze(0)
                    
                    # Apply temperature for exploration
                    if temperature > 0:
                        policy = torch.softmax(policy_logits / temperature, dim=0).cpu().numpy()
                        action = np.random.choice(self.action_space_size, p=policy)
                    else:
                        action = policy_logits.argmax().item()
                    
                    action_probs = torch.softmax(policy_logits, dim=0).cpu().numpy()
                    value = output['value'].item()
            
            action_counts[action] += 1
            
            # Store in game history (before taking action)
            # Store preprocessed observation as numpy array
            game.store(
                observation=obs_tensor.cpu().numpy(),  # Store preprocessed version
                action=action,
                reward=0,  # Will be updated after step
                policy=action_probs,
                value=value
            )
            
            # Take action in environment
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update reward in game history
            if len(game.rewards) > 0:
                game.rewards[-1] = reward
            
            if render:
                self.env.render()
        
        # Print action distribution for this game
        print(f"  Self-play game completed: {len(game)} steps, reward={sum(game.rewards):.1f}")
        print(f"    Action distribution: NOOP={action_counts[0]}, FIRE={action_counts[1]}, RIGHT={action_counts[2]}, LEFT={action_counts[3]}")
        
        # Save game to replay buffer
        self.replay_buffer.save_game(game)
        self.games_played += 1
        
        return game
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step
        
        Returns:
            Dictionary with training metrics
        """
        if not self.replay_buffer.is_ready():
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample_batch()
        
        # Move to device
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        target_values = batch['target_values'].to(self.device)
        target_rewards = batch['target_rewards'].to(self.device)
        target_policies = batch['target_policies'].to(self.device)
        weights = batch['weights'].to(self.device)
        
        # Forward pass through network
        outputs = self.network(observations, actions.unbind(1))
        
        # Compute losses
        losses = self.compute_losses(
            outputs, target_values, target_rewards, target_policies, weights
        )
        
        # Backward pass
        total_loss = losses['total_loss']
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % 100 == 0:
            self.update_target_network()
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': losses['value_loss'].item(),
            'reward_loss': losses['reward_loss'].item(),
            'policy_loss': losses['policy_loss'].item()
        }
    
    def compute_losses(self, outputs: Dict, target_values: torch.Tensor,
                      target_rewards: torch.Tensor, target_policies: torch.Tensor,
                      weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MuZero losses"""
        value_loss = 0
        reward_loss = 0
        policy_loss = 0
        
        # Process each unroll step
        num_unroll_steps = len(outputs['value']) - 1  # -1 because first step has no reward
        
        for i in range(num_unroll_steps + 1):
            # Value loss (categorical cross-entropy if using categorical)
            if self.config.use_categorical:
                target_value_cat = scalar_to_support(
                    target_values[:, i], self.config.support_size
                ).to(self.device)
                value_loss += F.cross_entropy(
                    outputs['value_logits'][i], 
                    target_value_cat.argmax(dim=1)
                )
            else:
                pred_value = outputs['value'][i]
                value_loss += F.mse_loss(pred_value, target_values[:, i])
            
            # Policy loss (cross-entropy)
            policy_loss += F.cross_entropy(
                outputs['policy_logits'][i],
                target_policies[:, i].argmax(dim=1)
            )
            
            # Reward loss (skip first step)
            if i > 0:
                if self.config.use_categorical:
                    target_reward_cat = scalar_to_support(
                        target_rewards[:, i], self.config.support_size
                    ).to(self.device)
                    reward_loss += F.cross_entropy(
                        outputs['reward_logits'][i],
                        target_reward_cat.argmax(dim=1)
                    )
                else:
                    pred_reward = outputs['reward'][i]
                    reward_loss += F.mse_loss(pred_reward, target_rewards[:, i])
        
        # Scale losses
        scale = 1.0 / (num_unroll_steps + 1)
        value_loss *= scale
        policy_loss *= scale
        if num_unroll_steps > 0:
            reward_loss *= scale
        
        # Apply importance sampling weights
        value_loss = (value_loss * weights).mean()
        policy_loss = (policy_loss * weights).mean()
        reward_loss = (reward_loss * weights).mean()
        
        # Total loss
        total_loss = value_loss + policy_loss + reward_loss
        
        return {
            'total_loss': total_loss,
            'value_loss': value_loss,
            'reward_loss': reward_loss,
            'policy_loss': policy_loss
        }
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.network.state_dict())
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < self.config.max_moves:
                # Preprocess observation
                obs_tensor = self.preprocess_observation(observation).to(self.device)
                
                # Run MCTS with no exploration
                mcts_result = self.mcts.run(
                    obs_tensor,
                    self.network,
                    temperature=0,  # Deterministic
                    add_exploration_noise=False
                )
                
                # Take action
                observation, reward, terminated, truncated, _ = self.env.step(mcts_result['action'])
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(episode_lengths),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'games_played': self.games_played,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.games_played = checkpoint['games_played']
        self.update_target_network()
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self, num_iterations: int = 1000000):
        """
        Main training loop (iteration-based)
        
        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(num_iterations):
            # Self-play
            if iteration % 10 == 0:  # Self-play every 10 iterations
                game = self.self_play()
                game_reward = sum(game.rewards)
                print(f"Game {self.games_played}: Reward = {game_reward:.1f}, Length = {len(game)}")
            
            # Training
            metrics = self.train_step()
            
            # Logging
            if iteration % self.config.log_interval == 0 and metrics:
                print(f"Step {self.training_steps}: Loss = {metrics['total_loss']:.4f}, "
                      f"Value = {metrics['value_loss']:.4f}, "
                      f"Policy = {metrics['policy_loss']:.4f}, "
                      f"Reward = {metrics['reward_loss']:.4f}")
            
            # Evaluation
            if iteration % self.config.test_interval == 0 and iteration > 0:
                eval_metrics = self.evaluate(self.config.test_episodes)
                print(f"Evaluation at step {self.training_steps}:")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
                print(f"  Max reward: {eval_metrics['max_reward']:.1f}")
            
            # Checkpoint
            if iteration % self.config.checkpoint_interval == 0 and iteration > 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"muzero_checkpoint_{self.training_steps}.pth"
                )
                self.save_checkpoint(checkpoint_path)
    
    def train_episodes(self, num_episodes: int = 100, train_steps_per_episode: int = 50):
        """
        Main training loop (episode-based)
        
        Args:
            num_episodes: Number of episodes to train
            train_steps_per_episode: Number of training steps per episode
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Self-play for one episode
            game = self.self_play()
            game_reward = sum(game.rewards)
            episode_rewards.append(game_reward)
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {game_reward:.1f}, Length = {len(game)}, "
                  f"Buffer size = {len(self.replay_buffer)}")
            
            # Training steps after each episode
            if self.replay_buffer.is_ready():
                losses = []
                print(f"  Starting {train_steps_per_episode} training steps...")
                for step in range(train_steps_per_episode):
                    metrics = self.train_step()
                    if metrics:
                        losses.append(metrics['total_loss'])
                        if (step + 1) % 5 == 0:  # Log every 5 steps
                            print(f"    Training step {step + 1}/{train_steps_per_episode}, loss = {metrics['total_loss']:.4f}")
                
                if losses:
                    avg_loss = np.mean(losses)
                    print(f"  Training complete: Avg loss = {avg_loss:.4f}")
            else:
                print(f"  Skipping training - buffer not ready (size: {len(self.replay_buffer)})")
            
            # Periodic evaluation
            if (episode + 1) % 10 == 0:
                print(f"\n--- Evaluation at episode {episode + 1} ---")
                eval_metrics = self.evaluate(min(5, self.config.test_episodes))
                print(f"  Mean reward: {eval_metrics['mean_reward']:.1f} ± {eval_metrics['std_reward']:.1f}")
                print(f"  Max reward: {eval_metrics['max_reward']:.1f}")
                
                # Show training progress
                recent_rewards = episode_rewards[-10:]
                print(f"  Recent training rewards (last {len(recent_rewards)} episodes): "
                      f"{np.mean(recent_rewards):.1f} ± {np.std(recent_rewards):.1f}")
                print("-" * 50)
            
            # Checkpoint every 20 episodes
            if (episode + 1) % 20 == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"muzero_checkpoint_ep{episode + 1}.pth"
                )
                self.save_checkpoint(checkpoint_path)
        
        # Final statistics
        print(f"\n{'='*60}")
        print(f"Training Complete - {num_episodes} Episodes")
        print(f"{'='*60}")
        print(f"Final Statistics:")
        print(f"  Total games played: {self.games_played}")
        print(f"  Total training steps: {self.training_steps}")
        print(f"  Mean episode reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"  Best episode reward: {np.max(episode_rewards):.1f}")
        print(f"  Worst episode reward: {np.min(episode_rewards):.1f}")
        
        # Show learning curve
        if len(episode_rewards) >= 10:
            print(f"\nLearning Progress:")
            chunks = 5
            chunk_size = len(episode_rewards) // chunks
            for i in range(chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < chunks - 1 else len(episode_rewards)
                chunk_rewards = episode_rewards[start:end]
                print(f"  Episodes {start+1}-{end}: {np.mean(chunk_rewards):.1f} ± {np.std(chunk_rewards):.1f}")