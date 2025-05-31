# cartpole/tests/test_policy.py
import unittest
import torch
import numpy as np
from cartpole.policy import PolicyNetwork

class TestPolicyNetwork(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 1
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.test_state_np_single = np.random.rand(self.state_dim).astype(np.float32)
        self.test_state_tensor_single = torch.from_numpy(self.test_state_np_single).unsqueeze(0)

    def test_initialization(self):
        self.assertIsNotNone(self.policy.fc1)
        self.assertIsNotNone(self.policy.fc_mean)
        self.assertIsNotNone(self.policy.fc_log_std)

    def test_forward_pass_single_state(self):
        mean, std = self.policy.forward(self.test_state_tensor_single)
        self.assertEqual(mean.shape, (1, self.action_dim))
        self.assertEqual(std.shape, (1, self.action_dim))
        self.assertTrue(torch.all(std > 0), "Standard deviation should be positive")
        self.assertTrue(torch.all(mean >= -1) and torch.all(mean <= 1), "Mean should be within [-1, 1] due to tanh")

    def test_select_action_stochastic_single_state(self):
        action_np, log_prob = self.policy.select_action(self.test_state_np_single, deterministic=False)

        self.assertIsInstance(action_np, np.ndarray, "Action should be a NumPy array")
        self.assertEqual(action_np.shape, (self.action_dim,), f"Action shape was {action_np.shape}")
        self.assertIsInstance(log_prob, torch.Tensor, "Log probability should be a torch Tensor")
        # For single state, single action_dim, log_prob from Normal(mean,std).log_prob().sum() will be shape (1,)
        self.assertEqual(log_prob.shape, (1,), f"Log_prob shape was {log_prob.shape}")


    def test_select_action_deterministic_single_state(self):
        action_np, log_prob = self.policy.select_action(self.test_state_np_single, deterministic=True)
        mean_tensor, _ = self.policy.forward(self.test_state_tensor_single)

        self.assertIsInstance(action_np, np.ndarray)
        self.assertEqual(action_np.shape, (self.action_dim,))
        self.assertAlmostEqual(action_np[0], mean_tensor.squeeze().item(), places=6) # Compare with the mean
        # log_prob is None for deterministic action as per implementation
        self.assertIsNone(log_prob, "Log_prob should be None for deterministic action")

    def test_forward_pass_batch_state(self):
        batch_size = 3
        batch_state_np = np.random.rand(batch_size, self.state_dim).astype(np.float32)
        batch_state_tensor = torch.from_numpy(batch_state_np)

        mean, std = self.policy.forward(batch_state_tensor)
        self.assertEqual(mean.shape, (batch_size, self.action_dim))
        self.assertEqual(std.shape, (batch_size, self.action_dim))
        self.assertTrue(torch.all(std > 0))
        self.assertTrue(torch.all(mean >= -1) and torch.all(mean <= 1))

    def test_select_action_stochastic_batch_state(self):
        batch_size = 3
        batch_state_np = np.random.rand(batch_size, self.state_dim).astype(np.float32)

        actions_np, log_probs = self.policy.select_action(batch_state_np, deterministic=False)
        self.assertIsInstance(actions_np, np.ndarray)
        self.assertEqual(actions_np.shape, (batch_size, self.action_dim))
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size,))

    def test_select_action_deterministic_batch_state(self):
        batch_size = 3
        batch_state_np = np.random.rand(batch_size, self.state_dim).astype(np.float32)

        actions_np_det, log_probs_det = self.policy.select_action(batch_state_np, deterministic=True)
        mean_tensor_det, _ = self.policy.forward(torch.from_numpy(batch_state_np))

        self.assertIsInstance(actions_np_det, np.ndarray)
        self.assertEqual(actions_np_det.shape, (batch_size, self.action_dim))
        self.assertTrue(np.allclose(actions_np_det, mean_tensor_det.detach().cpu().numpy(), atol=1e-6))
        self.assertIsNone(log_probs_det)


if __name__ == '__main__':
    unittest.main()
