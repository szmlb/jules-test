# cartpole/tests/test_cartpole_env.py
import unittest
import numpy as np
import math
from cartpole.cartpole_env import CartPoleEnv

class TestCartPoleEnv(unittest.TestCase):

    def setUp(self):
        self.env = CartPoleEnv()

    def test_initialization(self):
        self.assertIsNotNone(self.env.state)
        self.assertEqual(self.env.state.shape, (4,))
        self.assertTrue(all(abs(s) <= 0.05 for s in self.env.state), "Initial state not within [-0.05, 0.05]")
        # Check some physical parameters if they are public, or trust they are set
        self.assertEqual(self.env.gravity, 9.8)
        self.assertEqual(self.env.masscart, 1.0)
        self.assertEqual(self.env.masspole, 0.1)
        self.assertEqual(self.env.length, 0.5) # This is half_pole_length
        self.assertEqual(self.env.force_mag, 10.0)
        self.assertEqual(self.env.tau, 0.02)
        self.assertEqual(self.env.theta_threshold_radians, 12 * 2 * math.pi / 360)
        self.assertEqual(self.env.x_threshold, 2.4)

    def test_reset(self):
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)
        self.assertEqual(initial_state.shape, (4,))
        self.assertTrue(all(abs(s) <= 0.05 for s in initial_state))
        self.assertTrue(np.array_equal(self.env.state, initial_state))

    def test_step_basic_action(self):
        # Test a small action and observe state change
        initial_state = self.env.reset()
        # The CartPoleEnv step method expects a single float or a list/array with one float for the action
        action = np.array([1.0], dtype=np.float32)
        next_state, reward, done, _ = self.env.step(action)

        self.assertEqual(next_state.shape, (4,))
        self.assertFalse(np.array_equal(next_state, initial_state), "State should change after a step.")

        # Basic physics plausibility (without exact calculation)
        # If pole is upright and we push right (positive force), cart should move right.
        # With Euler integration: x_new = x_old + tau * x_dot_old.
        # If x_dot_old is 0, x_new will be x_old in the first step. Position changes in the second step.

        # Test push right
        self.env.reset()
        self.env.state = np.array([0, 0, 0, 0], dtype=np.float32) # Start from known state: x, x_dot, theta, theta_dot

        # Step 1
        next_state_step1, _, _, _ = self.env.step(np.array([1.0], dtype=np.float32)) # Push right
        self.assertAlmostEqual(next_state_step1[0], 0.0, places=5, msg="Cart position x should be ~0 after 1st step if x_dot was 0")
        self.assertTrue(next_state_step1[1] > 0, "Cart velocity x_dot should be > 0 after 1st step")

        # Step 2 (state from step1 is used, x_dot is now positive)
        next_state_step2, _, _, _ = self.env.step(np.array([0.0], dtype=np.float32)) # Continue with no additional force
        self.assertTrue(next_state_step2[0] > 0, "Cart position x should be > 0 after 2nd step")

        # Test push left
        self.env.reset()
        self.env.state = np.array([0, 0, 0, 0], dtype=np.float32)
        # Step 1
        next_state_step1_left, _, _, _ = self.env.step(np.array([-1.0], dtype=np.float32)) # Push left
        self.assertAlmostEqual(next_state_step1_left[0], 0.0, places=5, msg="Cart position x should be ~0 after 1st step if x_dot was 0 (left push)")
        self.assertTrue(next_state_step1_left[1] < 0, "Cart velocity x_dot should be < 0 after 1st step (left push)")

        # Step 2
        next_state_step2_left, _, _, _ = self.env.step(np.array([0.0], dtype=np.float32)) # Continue with no additional force
        self.assertTrue(next_state_step2_left[0] < 0, "Cart position x should be < 0 after 2nd step (left push)")

    def test_step_reward_and_done(self):
        self.env.reset()
        self.env.state = np.array([0, 0, 0, 0], dtype=np.float32) # Stable state

        # Not done
        _, reward, done, _ = self.env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)

        # Pole falls
        self.env.state = np.array([0, 0, self.env.theta_threshold_radians * 0.9, 0.1], dtype=np.float32)
        # One step that keeps it within limits
        _, reward, done, _ = self.env.step(np.array([0.01], dtype=np.float32))
        self.assertEqual(reward, 1.0, "Should get +1 reward if not done")
        self.assertFalse(done, "Should not be done if pole is just within limits")

        # Test done by pole angle
        self.env.state = np.array([0, 0, self.env.theta_threshold_radians + 0.01, 0], dtype=np.float32)
        _, reward, done, _ = self.env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(reward, 0.0, "Reward should be 0 if done")
        self.assertTrue(done, "Episode should be done if pole angle exceeds threshold")

        # Test done by cart position
        self.env.state = np.array([self.env.x_threshold + 0.1, 0, 0, 0], dtype=np.float32)
        _, reward, done, _ = self.env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done, "Episode should be done if cart position exceeds threshold")

        self.env.state = np.array([-self.env.x_threshold - 0.1, 0, 0, 0], dtype=np.float32)
        _, reward, done, _ = self.env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done, "Episode should be done if cart position is below -threshold")

    def test_action_clipping(self):
        self.env.reset()
        self.env.state = np.array([0,0,0,0], dtype=np.float32)
        # Action is clipped to [-force_mag, force_mag]
        # Send an action larger than force_mag
        large_force_action = np.array([self.env.force_mag + 10.0], dtype=np.float32)
        state_plus_force_mag, _, _, _ = self.env.step(np.array([self.env.force_mag], dtype=np.float32))
        state_large_force, _, _, _ = self.env.step(large_force_action)

        # Check if the effect of force_mag and force_mag + 10.0 are the same
        # This is an indirect way to check clipping.
        # We need to reset state for fair comparison
        self.env.state = np.array([0,0,0,0], dtype=np.float32)
        state_plus_force_mag, _, _, _ = self.env.step(np.array([self.env.force_mag], dtype=np.float32))
        self.env.state = np.array([0,0,0,0], dtype=np.float32) # Reset state again
        state_large_force, _, _, _ = self.env.step(large_force_action)

        self.assertTrue(np.allclose(state_plus_force_mag, state_large_force, atol=1e-6),
                        "Effect of action > force_mag should be same as action = force_mag due to clipping.")

    def test_render_and_close(self):
        # Basic check to see if render runs without error
        try:
            self.env.render(mode='human')
            self.assertIsNotNone(self.env.visualizer, "Visualizer should be created on render")
            self.env.close()
            self.assertIsNone(self.env.visualizer, "Visualizer should be None after close")
        except Exception as e:
            if "TclError" in str(e) or "no display name" in str(e) or "_tkinter.TclError" in str(e) or "Failed to import any qt binding" in str(e):
                print(f"Skipping render test due to display/GUI library error: {e}")
                self.skipTest(f"Skipping render test due to display/GUI library error: {e}")
            else:
                raise e

if __name__ == '__main__':
    unittest.main()
