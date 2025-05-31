# cartpole/tests/test_agent.py
import unittest
import torch
import numpy as np
from cartpole.agent import Agent
from cartpole.policy import PolicyNetwork

class TestAgent(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 1
        # Use small learning rate for tests to make parameter changes more subtle if not intended
        self.agent = Agent(self.state_dim, self.action_dim, lr=1e-4, gamma=0.99)

    def test_initialization(self):
        self.assertIsInstance(self.agent.policy, PolicyNetwork)
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.rewards, [])
        self.assertEqual(self.agent.log_probs, [])

    def test_select_action_stochastic(self):
        state_np = np.random.rand(self.state_dim).astype(np.float32)
        action = self.agent.select_action(state_np, deterministic=False)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertEqual(len(self.agent.log_probs), 1, "Log_prob should be stored for stochastic action")
        self.assertIsInstance(self.agent.log_probs[0], torch.Tensor)

        # Clear log_probs for next test
        self.agent.log_probs = []

    def test_select_action_deterministic(self):
        state_np = np.random.rand(self.state_dim).astype(np.float32)
        action = self.agent.select_action(state_np, deterministic=True)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertEqual(len(self.agent.log_probs), 0, "Log_prob should NOT be stored for deterministic action")

    def test_store_reward(self):
        self.agent.store_reward(1.0)
        self.agent.store_reward(0.5)
        self.assertEqual(self.agent.rewards, [1.0, 0.5])
        # Clear rewards for next test
        self.agent.rewards = []


    def test_update_policy_no_data(self):
        # Test if it handles an episode with no log_probs (e.g. all deterministic actions)
        # or no rewards.
        initial_params = [p.clone().detach() for p in self.agent.policy.parameters()]

        self.agent.rewards = [1.0, 1.0] # Has rewards
        self.agent.log_probs = []       # But no log_probs (e.g. if only deterministic actions were taken)
        self.agent.update_policy()

        for p_initial, p_updated in zip(initial_params, self.agent.policy.parameters()):
            self.assertTrue(torch.equal(p_initial, p_updated), "Policy should not change if no log_probs")
        self.assertEqual(self.agent.rewards, [], "Rewards should be cleared even if no update happened")
        self.assertEqual(self.agent.log_probs, [])

        # Test with no rewards (should also not update)
        self.agent.rewards = []
        # Add a dummy log_prob to see if it gets cleared
        self.agent.log_probs = [torch.tensor(0.1)]
        self.agent.update_policy()
        for p_initial, p_updated in zip(initial_params, self.agent.policy.parameters()):
            self.assertTrue(torch.equal(p_initial, p_updated), "Policy should not change if no rewards (even with log_probs)")
        self.assertEqual(self.agent.rewards, [])
        self.assertEqual(self.agent.log_probs, [], "Log_probs should be cleared")


    def test_update_policy_simple_episode(self):
        original_policy_params = [p.clone().detach() for p in self.agent.policy.parameters()]

        # Simulate a short episode
        state1 = np.random.rand(self.state_dim).astype(np.float32)
        state2 = np.random.rand(self.state_dim).astype(np.float32)

        # Step 1
        _ = self.agent.select_action(state1, deterministic=False)
        self.agent.store_reward(1.0)

        # Step 2
        _ = self.agent.select_action(state2, deterministic=False)
        self.agent.store_reward(0.0) # Changed reward to 0 to ensure returns are not all same

        self.assertEqual(len(self.agent.rewards), 2)
        self.assertEqual(len(self.agent.log_probs), 2)

        self.agent.update_policy()

        params_updated = False
        for p_orig, p_updated in zip(original_policy_params, self.agent.policy.parameters()):
            if not torch.equal(p_orig, p_updated):
                params_updated = True
                break
        self.assertTrue(params_updated, "Policy parameters should change after update_policy with data.")

        self.assertEqual(self.agent.rewards, [], "Rewards buffer should be cleared")
        self.assertEqual(self.agent.log_probs, [], "Log_probs buffer should be cleared")

    def test_update_policy_single_step_episode(self):
        original_policy_params = [p.clone().detach() for p in self.agent.policy.parameters()]
        state1 = np.random.rand(self.state_dim).astype(np.float32)
        _ = self.agent.select_action(state1, deterministic=False)
        self.agent.store_reward(1.0)

        self.agent.update_policy()

        params_updated = False
        for p_orig, p_updated in zip(original_policy_params, self.agent.policy.parameters()):
            if not torch.equal(p_orig, p_updated):
                params_updated = True
                break
        self.assertTrue(params_updated, "Policy parameters should change even for a single step episode.")
        self.assertEqual(self.agent.rewards, [])
        self.assertEqual(self.agent.log_probs, [])


if __name__ == '__main__':
    unittest.main()
