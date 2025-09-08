"""Unit tests for MADDPG Tennis implementation."""

import unittest
import numpy as np
import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMADDPGComponents(unittest.TestCase):
    """Test cases for MADDPG components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 24
        self.action_size = 2
        self.num_agents = 2
        self.seed = 42
        
    def test_actor_network_output_shape(self):
        """Test Actor network output shape."""
        from maddpg_agent import Actor
        
        actor = Actor(self.state_size, self.action_size, self.seed)
        state = torch.randn(self.state_size)
        action = actor(state)
        
        self.assertEqual(action.shape, torch.Size([1, self.action_size]))
        
    def test_action_bounds(self):
        """Test that actor outputs are bounded between -1 and 1."""
        from maddpg_agent import Actor
        
        actor = Actor(self.state_size, self.action_size, self.seed)
        state = torch.randn(self.state_size)
        action = actor(state)
        
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))


if __name__ == '__main__':
    unittest.main()
