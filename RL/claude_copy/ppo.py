# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from networks import ActorCritic
from replay_buffer import RolloutStorage
from envs import make_vec_envs
from utils import save_checkpoint, Logger, init_weights
from config import PPOConfig

class PPO:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Environments
        self.envs = make_vec_envs(
            config.env_name, 
            config.num_processes, 
            config.seed,
            config.frame_stack
        )
        
        # Network
        self.actor_critic = ActorCritic(
            config.frame_stack,
            self.envs.action_space.n
        ).to(self.device)
        self.actor_critic.apply(init_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.lr,
            eps=config.eps
        )
        
        # Storage
        self.rollouts = RolloutStorage(
            config.num_steps,
            config.num_processes,
            self.envs.observation_space.shape,
            self.envs.action_space
        )
        
        # Logger
        self.logger = Logger(config.log_dir)
        
        # Initialize
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(torch.FloatTensor(obs))
        
        self.episode_rewards = []
        self.episode_lengths = []
        
    def compute_action(self, obs, deterministic=False):
        """Compute action and value"""
        with torch.no_grad():
            logits, value = self.actor_critic(obs)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = probs.multinomial(1).squeeze(1)
            
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        
        return value, action, action_log_probs
    
    def update(self):
        """Update policy using PPO"""
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                self.rollouts.obs[-1]
            )
        
        self.rollouts.compute_returns(
            next_value, 
            self.config.use_gae,
            self.config.gamma,
            self.config.gae_lambda
        )
        
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        for e in range(self.config.ppo_epoch):
            data_generator = self.rollouts.feed_forward_generator(
                advantages, self.config.num_steps * self.config.num_processes // self.config.batch_size
            )
            
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                    masks_batch, old_action_log_probs_batch, adv_targ = sample
                
                # Forward pass
                logits, values = self.actor_critic(obs_batch)
                
                # Action loss
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions_batch).squeeze(1)
                
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch.squeeze())
                surr1 = ratio * adv_targ.squeeze()
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * adv_targ.squeeze()
                action_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.config.clip_param, self.config.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Entropy
                probs = F.softmax(logits, dim=-1)
                dist_entropy = -(log_probs * probs).sum(dim=-1).mean()
                
                # Total loss
                loss = action_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * dist_entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        num_updates = self.config.ppo_epoch * (self.config.num_steps * self.config.num_processes // self.config.batch_size)
        
        return value_loss_epoch / num_updates, action_loss_epoch / num_updates, dist_entropy_epoch / num_updates
    
    def train(self):
        """Main training loop"""
        episode_rewards = []
        episode_lengths = []
        episode_count = 0
        
        for update in range(self.config.num_env_steps // (self.config.num_steps * self.config.num_processes)):
            # Collect rollouts
            for step in range(self.config.num_steps):
                # Sample actions
                value, action, action_log_prob = self.compute_action(
                    self.rollouts.obs[step].to(self.device)
                )
                
                # Environment step
                obs, reward, done, infos = self.envs.step(action.cpu().numpy())
                
                # Track episodes
                for i, info in enumerate(infos):
                    if 'episode' in info:
                        episode_rewards.append(info['episode']['r'])
                        episode_lengths.append(info['episode']['l'])
                        self.logger.log_episode(info['episode']['r'], info['episode']['l'])
                        episode_count += 1
                
                # Create masks
                masks = torch.FloatTensor([[0.0] if done else [1.0] for done in done])
                
                # Insert to storage
                self.rollouts.insert(
                    torch.FloatTensor(obs),
                    action,
                    action_log_prob,
                    value,
                    torch.FloatTensor(reward).unsqueeze(1),
                    masks
                )
            
            # Update policy
            value_loss, action_loss, dist_entropy = self.update()
            
            # After update
            self.rollouts.after_update()
            
            # Logging
            if update % self.config.log_interval == 0 and len(episode_rewards) > 0:
                total_steps = (update + 1) * self.config.num_processes * self.config.num_steps
                stats = {
                    'mean_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
                    'value_loss': value_loss,
                    'action_loss': action_loss,
                    'entropy': dist_entropy,
                    'episodes': episode_count,
                }
                self.logger.log_metrics(total_steps, stats)
            
            # Save checkpoint
            if update % self.config.save_interval == 0:
                checkpoint = {
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'update': update,
                    'episode': episode_count,
                }
                save_checkpoint(checkpoint, os.path.join(self.config.save_dir, f'ppo_{update}.pth'))
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent"""
        eval_envs = make_vec_envs(
            self.config.env_name,
            1,
            self.config.seed + 1000,
            self.config.frame_stack
        )
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            obs = eval_envs.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    _, action, _ = self.compute_action(
                        torch.FloatTensor(obs).to(self.device),
                        deterministic=True
                    )
                
                obs, reward, done, _ = eval_envs.step(action.cpu().numpy())
                episode_reward += reward[0]
                done = done[0]
            
            eval_rewards.append(episode_reward)
        
        eval_envs.close()
        return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    config = PPOConfig()
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create and train agent
    agent = PPO(config)
    agent.train()
    
    # Evaluate final performance
    mean_reward, std_reward = agent.evaluate()
    print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()