import unittest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add current directory to path for imports when run from RL folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.distributed_buffer import DistributedReplayBuffer, DistributedFrameStack


class TestDistributedReplayBuffer(unittest.TestCase):
    """Test cases for DistributedReplayBuffer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.buffer_capacity = 1000
        self.buffer = DistributedReplayBuffer(self.buffer_capacity)
        
        # Create sample experience data
        self.sample_state = np.random.rand(4, 84, 84)
        self.sample_action = 0
        self.sample_reward = 1.0
        self.sample_next_state = np.random.rand(4, 84, 84)
        self.sample_done = False
        self.sample_worker_id = 0
    
    def test_buffer_initialization(self):
        """Test buffer is properly initialized"""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertIsNotNone(self.buffer.lock)
        self.assertEqual(len(self.buffer._episode_boundaries), 0)
    
    def test_single_experience_push(self):
        """Test pushing a single experience"""
        self.buffer.push(
            self.sample_state, self.sample_action, self.sample_reward,
            self.sample_next_state, self.sample_done, self.sample_worker_id
        )
        
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(len(self.buffer._episode_boundaries), 0)  # Not done
    
    def test_episode_boundary_tracking(self):
        """Test that episode boundaries are tracked correctly"""
        # Add experiences including one that ends an episode
        for i in range(5):
            done = (i == 4)  # Last experience ends episode
            self.buffer.push(
                self.sample_state, self.sample_action, self.sample_reward,
                self.sample_next_state, done, self.sample_worker_id
            )
        
        self.assertEqual(len(self.buffer), 5)
        self.assertEqual(len(self.buffer._episode_boundaries), 1)
        self.assertEqual(self.buffer._episode_boundaries[0], 4)  # Index of done experience
    
    def test_batch_push(self):
        """Test pushing multiple experiences at once"""
        experiences = []
        for i in range(10):
            done = (i % 5 == 4)  # Every 5th experience ends episode
            exp = (self.sample_state, self.sample_action, self.sample_reward,
                   self.sample_next_state, done, i % 3)  # Different worker IDs
            experiences.append(exp)
        
        self.buffer.push_batch(experiences)
        
        self.assertEqual(len(self.buffer), 10)
        self.assertEqual(len(self.buffer._episode_boundaries), 2)  # Two episodes completed
    
    def test_sample_basic(self):
        """Test basic sampling functionality"""
        # Fill buffer with some experiences
        for i in range(50):
            self.buffer.push(
                self.sample_state, i % 4, float(i), self.sample_next_state,
                False, i % 3
            )
        
        # Test sampling
        batch_size = 10
        sample = self.buffer.sample(batch_size)
        
        self.assertIsNotNone(sample)
        states, actions, rewards, next_states, dones, worker_ids = sample
        
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        self.assertEqual(len(worker_ids), batch_size)
    
    def test_sample_insufficient_data(self):
        """Test sampling when buffer has insufficient data"""
        # Add only 5 experiences
        for i in range(5):
            self.buffer.push(
                self.sample_state, 0, 1.0, self.sample_next_state, False, 0
            )
        
        # Try to sample more than available
        sample = self.buffer.sample(10)
        self.assertIsNone(sample)
    
    def test_thread_safety(self):
        """Test that buffer operations are thread-safe"""
        num_threads = 5
        experiences_per_thread = 100
        
        def worker_function(worker_id):
            """Function for each worker thread"""
            for i in range(experiences_per_thread):
                self.buffer.push(
                    self.sample_state, i % 4, float(i), self.sample_next_state,
                    i % 20 == 19, worker_id  # Episode ends every 20 steps
                )
        
        # Start multiple threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        # Verify results
        expected_total = num_threads * experiences_per_thread
        self.assertEqual(len(self.buffer), expected_total)
        
        # Check that we can sample without errors
        sample = self.buffer.sample(32)
        self.assertIsNotNone(sample)
    
    def test_episode_sampling(self):
        """Test sampling complete episodes"""
        # Create a complete episode
        episode_length = 10
        for i in range(episode_length):
            done = (i == episode_length - 1)
            self.buffer.push(
                self.sample_state, i % 4, float(i), self.sample_next_state,
                done, 0
            )
        
        # Sample episode
        episode = self.buffer.sample_episode()
        
        self.assertIsNotNone(episode)
        self.assertEqual(episode['episode_length'], episode_length)
        self.assertEqual(episode['total_reward'], sum(range(episode_length)))
        self.assertEqual(len(episode['states']), episode_length)
    
    def test_statistics(self):
        """Test statistics gathering"""
        # Add experiences from multiple workers
        for worker_id in range(3):
            for i in range(20):
                done = (i == 19)
                self.buffer.push(
                    self.sample_state, 0, 1.0, self.sample_next_state,
                    done, worker_id
                )
        
        stats = self.buffer.get_statistics()
        
        self.assertEqual(stats['size'], 60)
        self.assertEqual(stats['capacity'], self.buffer_capacity)
        self.assertEqual(stats['num_episodes'], 3)
        self.assertIn('worker_distribution', stats)
        self.assertEqual(len(stats['worker_distribution']), 3)
    
    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is exceeded"""
        # Fill buffer beyond capacity
        overflow_size = int(self.buffer_capacity * 1.5)
        for i in range(overflow_size):
            self.buffer.push(
                self.sample_state, 0, float(i), self.sample_next_state,
                i % 50 == 49, 0  # Episode every 50 steps
            )
        
        # Buffer should not exceed capacity
        self.assertEqual(len(self.buffer), self.buffer_capacity)
        
        # Should still be able to sample
        sample = self.buffer.sample(32)
        self.assertIsNotNone(sample)
    
    def test_clear_buffer(self):
        """Test clearing the buffer"""
        # Add some experiences
        for i in range(10):
            self.buffer.push(
                self.sample_state, 0, 1.0, self.sample_next_state, False, 0
            )
        
        self.assertEqual(len(self.buffer), 10)
        
        # Clear buffer
        self.buffer.clear()
        
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer._episode_boundaries), 0)


class TestDistributedFrameStack(unittest.TestCase):
    """Test cases for DistributedFrameStack"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.frame_stack_size = 4
        self.frame_stack = DistributedFrameStack(self.frame_stack_size)
        self.sample_frame = np.random.rand(84, 84, 3)  # H, W, C format
    
    def test_initialization(self):
        """Test frame stack initialization"""
        self.assertEqual(self.frame_stack.num_stack, self.frame_stack_size)
        self.assertEqual(len(self.frame_stack.worker_frames), 0)
        self.assertIsNotNone(self.frame_stack.lock)
    
    def test_reset_single_worker(self):
        """Test resetting frame stack for single worker"""
        worker_id = 0
        stacked_obs = self.frame_stack.reset(self.sample_frame, worker_id)
        
        # Should have the worker in the dictionary
        self.assertIn(worker_id, self.frame_stack.worker_frames)
        
        # Should have correct number of frames
        self.assertEqual(len(self.frame_stack.worker_frames[worker_id]), self.frame_stack_size)
        
        # Output should have correct shape
        expected_shape = (self.frame_stack_size * 3, 84, 84)  # C*stack, H, W
        self.assertEqual(stacked_obs.shape, expected_shape)
    
    def test_append_frames(self):
        """Test appending frames to stack"""
        worker_id = 0
        
        # Reset first
        self.frame_stack.reset(self.sample_frame, worker_id)
        
        # Append a different frame
        new_frame = np.random.rand(84, 84, 3)
        stacked_obs = self.frame_stack.append(new_frame, worker_id)
        
        # Should still have correct shape
        expected_shape = (self.frame_stack_size * 3, 84, 84)
        self.assertEqual(stacked_obs.shape, expected_shape)
    
    def test_multiple_workers(self):
        """Test frame stacking with multiple workers"""
        num_workers = 4
        
        # Reset for all workers
        for worker_id in range(num_workers):
            frame = np.random.rand(84, 84, 3)
            self.frame_stack.reset(frame, worker_id)
        
        # Check all workers are tracked
        self.assertEqual(len(self.frame_stack.worker_frames), num_workers)
        
        # Append frames for each worker
        for worker_id in range(num_workers):
            new_frame = np.random.rand(84, 84, 3)
            stacked_obs = self.frame_stack.append(new_frame, worker_id)
            expected_shape = (self.frame_stack_size * 3, 84, 84)
            self.assertEqual(stacked_obs.shape, expected_shape)
    
    def test_thread_safety(self):
        """Test thread safety of frame stacking"""
        num_workers = 5
        frames_per_worker = 20
        
        def worker_function(worker_id):
            """Function for each worker thread"""
            # Reset
            frame = np.random.rand(84, 84, 3)
            self.frame_stack.reset(frame, worker_id)
            
            # Append frames
            for i in range(frames_per_worker):
                new_frame = np.random.rand(84, 84, 3)
                stacked_obs = self.frame_stack.append(new_frame, worker_id)
                
                # Verify shape
                expected_shape = (self.frame_stack_size * 3, 84, 84)
                self.assertEqual(stacked_obs.shape, expected_shape)
        
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_workers)]
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Verify all workers are tracked
        self.assertEqual(len(self.frame_stack.worker_frames), num_workers)
    
    def test_clear_worker(self):
        """Test clearing frames for specific worker"""
        worker_id = 0
        
        # Reset and verify worker exists
        self.frame_stack.reset(self.sample_frame, worker_id)
        self.assertIn(worker_id, self.frame_stack.worker_frames)
        
        # Clear worker
        self.frame_stack.clear_worker(worker_id)
        self.assertNotIn(worker_id, self.frame_stack.worker_frames)
    
    def test_append_without_reset(self):
        """Test appending to worker that hasn't been reset"""
        worker_id = 5
        
        # Append without reset (should auto-initialize)
        stacked_obs = self.frame_stack.append(self.sample_frame, worker_id)
        
        # Should have created the worker
        self.assertIn(worker_id, self.frame_stack.worker_frames)
        
        # Should have correct shape
        expected_shape = (self.frame_stack_size * 3, 84, 84)
        self.assertEqual(stacked_obs.shape, expected_shape)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)