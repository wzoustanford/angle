import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import Mock, patch

# Add current directory to path for imports when run from RL folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.distributed_buffer import DistributedReplayBuffer
from model.parallel_env_manager import EnvironmentWorker, ParallelEnvironmentManager
from config.AgentConfig import AgentConfig


class TestEnvironmentWorker(unittest.TestCase):
    """Test cases for EnvironmentWorker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.worker_id = 0
        self.env_name = 'ALE/SpaceInvaders-v5'
        self.frame_stack = 4
        self.replay_buffer = DistributedReplayBuffer(1000)
        
        # Create worker
        self.worker = EnvironmentWorker(
            worker_id=self.worker_id,
            env_name=self.env_name,
            frame_stack=self.frame_stack,
            replay_buffer=self.replay_buffer
        )
    
    def test_worker_initialization(self):
        """Test worker is properly initialized"""
        self.assertEqual(self.worker.worker_id, self.worker_id)
        self.assertEqual(self.worker.env_name, self.env_name)
        self.assertIsNotNone(self.worker.env)
        self.assertIsNotNone(self.worker.frame_stack)
        self.assertEqual(self.worker.n_actions, 6)  # Space Invaders has 6 actions
        self.assertEqual(self.worker.episode_count, 0)
        self.assertEqual(self.worker.total_steps, 0)
        self.assertEqual(len(self.worker.episode_rewards), 0)
    
    def test_action_selection_random(self):
        """Test random action selection"""
        state = np.random.rand(12, 210, 160)  # 4 frames * 3 channels
        
        # With no shared network, should always be random
        action = self.worker.select_action(state, epsilon=0.5)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.worker.n_actions)
    
    def test_action_selection_with_network(self):
        """Test action selection with shared network"""
        state = np.random.rand(12, 210, 160)
        
        # Mock a network
        mock_network = Mock()
        mock_q_values = Mock()
        mock_q_values.argmax.return_value.item.return_value = 2
        mock_network.return_value = mock_q_values
        
        self.worker.shared_network = mock_network
        
        # With epsilon=0, should use network
        action = self.worker.select_action(state, epsilon=0.0)
        self.assertEqual(action, 2)
        
        # Verify network was called
        mock_network.assert_called_once()
    
    def test_run_single_episode(self):
        """Test running a single episode"""
        initial_buffer_size = len(self.replay_buffer)
        
        # Run episode with short max_steps for testing
        result = self.worker.run_episode(max_steps=100, epsilon=1.0)  # Full random
        
        # Check result structure
        self.assertIn('worker_id', result)
        self.assertIn('episode_reward', result)
        self.assertIn('episode_length', result)
        self.assertIn('total_episodes', result)
        self.assertIn('total_steps', result)
        
        # Check values
        self.assertEqual(result['worker_id'], self.worker_id)
        self.assertEqual(result['total_episodes'], 1)
        self.assertGreater(result['episode_length'], 0)
        
        # Check that experiences were added to buffer
        self.assertGreater(len(self.replay_buffer), initial_buffer_size)
        
        # Check worker state updated
        self.assertEqual(self.worker.episode_count, 1)
        self.assertGreater(self.worker.total_steps, 0)
        self.assertEqual(len(self.worker.episode_rewards), 1)
    
    def test_worker_statistics(self):
        """Test worker statistics collection"""
        # Run a few episodes
        for _ in range(3):
            self.worker.run_episode(max_steps=50, epsilon=1.0)
        
        stats = self.worker.get_statistics()
        
        # Check statistics structure
        required_keys = ['worker_id', 'total_episodes', 'total_steps', 
                        'avg_reward', 'recent_avg_reward', 'max_reward', 'min_reward']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values
        self.assertEqual(stats['worker_id'], self.worker_id)
        self.assertEqual(stats['total_episodes'], 3)
        self.assertGreater(stats['total_steps'], 0)
    
    def test_worker_stop(self):
        """Test stopping worker"""
        # Start continuous run in background (would run forever)
        self.worker.running = True
        
        # Stop immediately
        self.worker.stop()
        
        self.assertFalse(self.worker.running)


class TestParallelEnvironmentManager(unittest.TestCase):
    """Test cases for ParallelEnvironmentManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig()
        self.replay_buffer = DistributedReplayBuffer(10000)
        self.num_workers = 2  # Use fewer workers for testing
        
        self.manager = ParallelEnvironmentManager(
            config=self.config,
            replay_buffer=self.replay_buffer,
            num_workers=self.num_workers
        )
    
    def tearDown(self):
        """Clean up after tests"""
        try:
            self.manager.stop_collection()
            self.manager.executor.shutdown(wait=False)
        except:
            pass
    
    def test_manager_initialization(self):
        """Test manager is properly initialized"""
        self.assertEqual(self.manager.num_workers, self.num_workers)
        self.assertEqual(len(self.manager.workers), self.num_workers)
        self.assertIsNotNone(self.manager.executor)
        self.assertFalse(self.manager.running)
        
        # Check workers are properly initialized
        for i, worker in enumerate(self.manager.workers):
            self.assertEqual(worker.worker_id, i)
            self.assertEqual(worker.env_name, self.config.env_name)
    
    def test_collect_batch(self):
        """Test batch collection functionality"""
        initial_buffer_size = len(self.replay_buffer)
        
        # Collect small batch for testing
        episodes_per_worker = 2
        results = self.manager.collect_batch(
            num_episodes_per_worker=episodes_per_worker,
            epsilon=1.0  # Full random for speed
        )
        
        # Check results
        expected_total_episodes = self.num_workers * episodes_per_worker
        self.assertEqual(len(results), expected_total_episodes)
        
        # Check that each result has required fields
        for result in results:
            self.assertIn('worker_id', result)
            self.assertIn('episode_reward', result)
            self.assertIn('episode_length', result)
        
        # Check that experiences were added to buffer
        self.assertGreater(len(self.replay_buffer), initial_buffer_size)
    
    def test_update_shared_network(self):
        """Test updating shared network"""
        # Mock network
        mock_network = Mock()
        
        # Update shared network
        self.manager.update_shared_network(mock_network)
        
        # Check that all workers have the network
        for worker in self.manager.workers:
            self.assertEqual(worker.shared_network, mock_network)
    
    def test_statistics_collection(self):
        """Test statistics aggregation"""
        # Run some episodes first
        self.manager.collect_batch(num_episodes_per_worker=1, epsilon=1.0)
        
        stats = self.manager.get_statistics()
        
        # Check statistics structure
        required_keys = ['num_workers', 'total_episodes', 'total_steps',
                        'worker_stats', 'replay_buffer_stats']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values
        self.assertEqual(stats['num_workers'], self.num_workers)
        self.assertGreater(stats['total_episodes'], 0)
        self.assertEqual(len(stats['worker_stats']), self.num_workers)
    
    def test_start_stop_collection(self):
        """Test starting and stopping continuous collection"""
        # Start collection
        self.manager.start_collection(episodes_per_worker=5)
        self.assertTrue(self.manager.running)
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop collection
        self.manager.stop_collection()
        self.assertFalse(self.manager.running)
        
        # Check that some work was done
        stats = self.manager.get_statistics()
        # Note: Due to timing, we might not always have episodes completed
        # So we don't assert on total_episodes being > 0
    
    def test_context_manager(self):
        """Test using manager as context manager"""
        initial_buffer_size = len(self.replay_buffer)
        
        with ParallelEnvironmentManager(
            config=self.config,
            replay_buffer=self.replay_buffer,
            num_workers=2
        ) as manager:
            # Use the manager
            results = manager.collect_batch(num_episodes_per_worker=1, epsilon=1.0)
            self.assertGreater(len(results), 0)
        
        # Manager should be properly cleaned up
        # Buffer should have new experiences
        self.assertGreater(len(self.replay_buffer), initial_buffer_size)


class TestIntegration(unittest.TestCase):
    """Integration tests for distributed components working together"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.config = AgentConfig()
        self.replay_buffer = DistributedReplayBuffer(5000)
        self.num_workers = 2
    
    def test_full_collection_pipeline(self):
        """Test the complete collection pipeline"""
        with ParallelEnvironmentManager(
            config=self.config,
            replay_buffer=self.replay_buffer,
            num_workers=self.num_workers
        ) as manager:
            
            # Collect experiences
            episodes_per_worker = 3
            results = manager.collect_batch(
                num_episodes_per_worker=episodes_per_worker,
                epsilon=1.0
            )
            
            # Verify collection worked
            self.assertEqual(len(results), self.num_workers * episodes_per_worker)
            
            # Check buffer has experiences
            self.assertGreater(len(self.replay_buffer), 0)
            
            # Test sampling from buffer
            if len(self.replay_buffer) >= 32:
                sample = self.replay_buffer.sample(32)
                self.assertIsNotNone(sample)
                
                states, actions, rewards, next_states, dones, worker_ids = sample
                self.assertEqual(len(states), 32)
                self.assertEqual(len(set(worker_ids)), min(self.num_workers, len(set(worker_ids))))
            
            # Test statistics
            manager_stats = manager.get_statistics()
            buffer_stats = self.replay_buffer.get_statistics()
            
            self.assertGreater(manager_stats['total_episodes'], 0)
            self.assertGreater(buffer_stats['size'], 0)
            self.assertIn('worker_distribution', buffer_stats)
    
    def test_concurrent_collection_and_sampling(self):
        """Test that collection and sampling can happen concurrently"""
        manager = ParallelEnvironmentManager(
            config=self.config,
            replay_buffer=self.replay_buffer,
            num_workers=self.num_workers
        )
        
        try:
            # Start collection
            manager.start_collection(episodes_per_worker=10)
            
            # Wait a bit for some experiences to be collected
            time.sleep(3)
            
            # Sample while collection is ongoing
            samples_collected = 0
            for _ in range(5):
                if len(self.replay_buffer) >= 32:
                    sample = self.replay_buffer.sample(32)
                    if sample is not None:
                        samples_collected += 1
                time.sleep(0.5)
            
            # Stop collection
            manager.stop_collection()
            
            # Should have been able to sample
            self.assertGreater(samples_collected, 0)
            
        finally:
            manager.stop_collection()
            manager.executor.shutdown(wait=False)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)