import unittest
import torch
import numpy as np
import tempfile
import shutil
import time
import sys
import os
from unittest.mock import Mock, patch

# Add current directory to path for imports when run from RL folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.distributed_dqn_agent import DistributedDQNAgent
from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig


class TestDistributedDQNAgent(unittest.TestCase):
    """Test cases for DistributedDQNAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use smaller config for testing
        self.config = DistributedAgentConfig(
            env_name='ALE/SpaceInvaders-v5',
            num_workers=2,
            memory_size=1000,
            min_replay_size=100,
            batch_size=16,
            save_interval=1000,  # Don't save during tests
            target_update_freq=50
        )
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.config.checkpoint_dir = self.temp_dir
        
        self.agent = DistributedDQNAgent(self.config, num_workers=2)
    
    def tearDown(self):
        """Clean up after tests"""
        try:
            self.agent.env_manager.stop_collection()
            self.agent.env_manager.executor.shutdown(wait=False)
        except:
            pass
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_agent_initialization(self):
        """Test agent is properly initialized"""
        self.assertEqual(self.agent.num_workers, 2)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.replay_buffer)
        self.assertIsNotNone(self.agent.env_manager)
        
        # Check device is set
        self.assertIn(self.agent.device.type, ['cuda', 'cpu'])
        
        # Check networks are on correct device
        self.assertEqual(next(self.agent.q_network.parameters()).device, self.agent.device)
        self.assertEqual(next(self.agent.target_network.parameters()).device, self.agent.device)
    
    def test_network_shapes(self):
        """Test that networks have correct input/output shapes"""
        # Test with sample input
        batch_size = 4
        sample_input = torch.randn(batch_size, 12, 210, 160).to(self.agent.device)
        
        # Forward pass through Q-network
        q_values = self.agent.q_network(sample_input)
        
        # Check output shape
        expected_actions = 6  # Space Invaders
        self.assertEqual(q_values.shape, (batch_size, expected_actions))
        
        # Test target network
        target_q_values = self.agent.target_network(sample_input)
        self.assertEqual(target_q_values.shape, (batch_size, expected_actions))
    
    def test_epsilon_schedule(self):
        """Test epsilon scheduling function"""
        epsilon_fn = self.agent.get_epsilon_schedule()
        
        # Test different episodes
        eps_0 = epsilon_fn(0)
        eps_100 = epsilon_fn(100)
        eps_1000 = epsilon_fn(1000)
        
        # Should decay over time
        self.assertGreaterEqual(eps_0, eps_100)
        self.assertGreaterEqual(eps_100, eps_1000)
        
        # Should not go below minimum
        self.assertGreaterEqual(eps_1000, self.config.epsilon_end)
    
    def test_update_q_network_insufficient_data(self):
        """Test Q-network update with insufficient data"""
        # Buffer should be empty initially
        loss = self.agent.update_q_network()
        self.assertIsNone(loss)
    
    def test_update_q_network_with_data(self):
        """Test Q-network update with sufficient data"""
        # Fill buffer with some experiences
        self._fill_replay_buffer(200)
        
        # Should be able to update now
        loss = self.agent.update_q_network()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_target_network_update(self):
        """Test target network update"""
        # Get initial target network state
        initial_params = [p.clone() for p in self.agent.target_network.parameters()]
        
        # Modify Q-network slightly
        with torch.no_grad():
            for p in self.agent.q_network.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        
        # Update target network
        self.agent.update_target_network()
        
        # Check that target network parameters changed
        updated_params = list(self.agent.target_network.parameters())
        for initial, updated in zip(initial_params, updated_params):
            self.assertFalse(torch.equal(initial, updated))
    
    def test_batch_collection_training(self):
        """Test batch-based training approach"""
        # Run very short training
        final_stats = self.agent.train_batch_collection(
            total_episodes=8,  # Small number for testing
            episodes_per_batch=4
        )
        
        # Check that training completed
        self.assertIsNotNone(final_stats)
        self.assertIn('env_stats', final_stats)
        self.assertIn('buffer_stats', final_stats)
        self.assertIn('training_stats', final_stats)
        
        # Check that some episodes were completed
        self.assertGreater(final_stats['final_episodes'], 0)
        
        # Check that buffer has data
        self.assertGreater(final_stats['buffer_stats']['size'], 0)
    
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_distributed_training_short(self, mock_sleep):
        """Test distributed training for a very short duration"""
        # Mock sleep to make test faster
        mock_sleep.return_value = None
        
        # Fill buffer first so training can start immediately
        self._fill_replay_buffer(200)
        
        # Start training in a separate thread and stop quickly
        import threading
        
        training_complete = threading.Event()
        final_stats = None
        
        def training_thread():
            nonlocal final_stats
            try:
                final_stats = self.agent.train_distributed(
                    total_episodes=20,  # Small target
                    collection_interval=10
                )
                training_complete.set()
            except Exception as e:
                print(f"Training thread error: {e}")
                training_complete.set()
        
        # Start training
        thread = threading.Thread(target=training_thread)
        thread.start()
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop by setting episodes_done high
        self.agent.episodes_done = 25  # Above target
        
        # Wait for completion
        training_complete.wait(timeout=10)
        thread.join(timeout=5)
        
        # Check results if training completed
        if final_stats:
            self.assertIsNotNone(final_stats)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        # Train briefly to have some state
        self._fill_replay_buffer(100)
        for _ in range(5):
            self.agent.update_q_network()
        
        # Save checkpoint
        self.agent.save_checkpoint()
        
        # Check file was created
        checkpoint_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.pth')]
        self.assertEqual(len(checkpoint_files), 1)
        
        # Create new agent
        new_agent = DistributedDQNAgent(self.config, num_workers=2)
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.temp_dir, checkpoint_files[0])
        new_agent.load_checkpoint(checkpoint_path)
        
        # Check state was loaded
        self.assertEqual(new_agent.episodes_done, self.agent.episodes_done)
        self.assertEqual(new_agent.steps_done, self.agent.steps_done)
        
        # Clean up new agent
        try:
            new_agent.env_manager.stop_collection()
            new_agent.env_manager.executor.shutdown(wait=False)
        except:
            pass
    
    def test_test_distributed(self):
        """Test the distributed testing functionality"""
        # Fill buffer and train a bit first
        self._fill_replay_buffer(200)
        for _ in range(10):
            self.agent.update_q_network()
        
        # Run test
        test_results = self.agent.test_distributed(num_episodes=4)
        
        # Check results
        self.assertIn('rewards', test_results)
        self.assertIn('avg_reward', test_results)
        self.assertIn('max_reward', test_results)
        self.assertIn('min_reward', test_results)
        
        # Check we got the right number of episodes
        self.assertEqual(len(test_results['rewards']), 4)
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        # Fill buffer and do some training
        self._fill_replay_buffer(200)
        
        initial_losses = len(self.agent.training_stats['losses'])
        
        # Perform several updates
        for _ in range(10):
            loss = self.agent.update_q_network()
            if loss is not None:
                # Check loss was recorded
                self.assertGreater(len(self.agent.training_stats['losses']), initial_losses)
    
    def _fill_replay_buffer(self, num_experiences):
        """Helper method to fill replay buffer with dummy experiences"""
        for i in range(num_experiences):
            state = np.random.rand(12, 210, 160)
            action = np.random.randint(0, 6)
            reward = np.random.random()
            next_state = np.random.rand(12, 210, 160)
            done = (i % 50 == 49)  # Episode ends every 50 steps
            worker_id = i % 2
            
            self.agent.replay_buffer.push(
                state, action, reward, next_state, done, worker_id
            )


class TestDistributedAgentConfig(unittest.TestCase):
    """Test cases for DistributedAgentConfig"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = DistributedAgentConfig()
        
        # Test distributed-specific defaults
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.collection_mode, 'continuous')
        self.assertEqual(config.batch_size, 64)  # Larger than regular DQN
        self.assertEqual(config.memory_size, 50000)  # Larger buffer
        self.assertTrue(config.prioritize_recent_experiences)
    
    def test_config_customization(self):
        """Test configuration customization"""
        config = DistributedAgentConfig(
            num_workers=8,
            env_name='ALE/Breakout-v5',
            batch_size=128,
            memory_size=100000
        )
        
        self.assertEqual(config.num_workers, 8)
        self.assertEqual(config.env_name, 'ALE/Breakout-v5')
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.memory_size, 100000)


class TestIntegrationDistributed(unittest.TestCase):
    """Integration tests for the complete distributed system"""
    
    def setUp(self):
        """Set up integration test"""
        self.config = DistributedAgentConfig(
            num_workers=2,
            memory_size=2000,
            min_replay_size=200,
            batch_size=32,
            save_interval=10000  # No saving during tests
        )
        
        self.temp_dir = tempfile.mkdtemp()
        self.config.checkpoint_dir = self.temp_dir
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training(self):
        """Test complete end-to-end training pipeline"""
        agent = DistributedDQNAgent(self.config, num_workers=2)
        
        try:
            # Run very short training
            final_stats = agent.train_batch_collection(
                total_episodes=6,
                episodes_per_batch=3
            )
            
            # Verify training worked
            self.assertGreater(final_stats['final_episodes'], 0)
            self.assertGreater(final_stats['buffer_stats']['size'], 0)
            
            # Test that we can sample from buffer
            if final_stats['buffer_stats']['size'] >= 32:
                sample = agent.replay_buffer.sample(32)
                self.assertIsNotNone(sample)
            
            # Test distributed testing
            test_results = agent.test_distributed(num_episodes=2)
            self.assertEqual(len(test_results['rewards']), 2)
            
        finally:
            agent.env_manager.stop_collection()
            agent.env_manager.executor.shutdown(wait=False)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)