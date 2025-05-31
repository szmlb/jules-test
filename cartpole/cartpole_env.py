import math
import numpy as np
from cartpole.utils.visualization import CartPoleVisualizer # Added import

class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Action space: A single continuous force value.
        # For simplicity, we'll assume the agent's output is directly the force.
        # We can define bounds if necessary, e.g., action_low = -self.force_mag, action_high = self.force_mag
        # For now, let's assume the policy will output values within a reasonable range.

        # Observation space:
        # Values are somewhat arbitrary, but cover a typical range.
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space_high = high
        self.observation_space_low = -high

        self.state = None
        self.visualizer = None # Added visualizer attribute
        self.reset()

    def step(self, action):
        # For continuous action, action is the force itself.
        # We might want to clip the action if the policy can output unbounded values.
        # For this example, let's assume the policy provides a sensible force.
        force = np.clip(action, -self.force_mag, self.force_mag)[0] # Assuming action is a single element array or list

        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Dynamics equations from OpenAI Gym classic_control cartpole.py,
        # which are derived from "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state using Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Done condition:
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        else:
            # Reward for the step that leads to termination.
            # Could also be 0 or a specific penalty.
            # For REINFORCE, the absence of future +1 rewards is the main driver.
            # Let's provide a neutral reward for the terminal step itself for now.
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        # Reset state to random values within a small range around the upright position
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = np.array(self.state, dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        if mode == 'human':
            if self.visualizer is None:
                self.visualizer = CartPoleVisualizer(
                    world_width=self.x_threshold * 2,
                    pole_length=self.length * 2 # Use the full pole length for visualization
                )

            if self.state is None:
                # Or raise an error: environment not reset yet.
                # For now, just return if state is not set (e.g. before first reset)
                return None

            cart_x = self.state[0]
            pole_theta = self.state[2]
            self.visualizer.render(cart_x, pole_theta)
            return self.visualizer.fig # Optionally return figure for other uses
        else:
            # Handle other modes if necessary, e.g. 'rgb_array'
            # For now, pass or raise error for unsupported modes.
            super(CartPoleEnv, self).render(mode=mode) # If inheriting from gym.Env

    def close(self):
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None

if __name__ == '__main__':
    # Example usage:
    env = CartPoleEnv()
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")

    # Simulate a few steps with random actions and rendering
    for i in range(50): # Increased steps for better visualization
        random_action = np.random.uniform(low=-1.0, high=1.0, size=(1,)) # Scaled action for policy
        # The policy outputs tanh, so action is in [-1,1]. Scale it by force_mag
        # Or, let the environment clip it. The current env clips action to [-force_mag, force_mag].
        # Let's provide action in the range expected by policy network for consistency in example.
        # The current env.step clips action, so this is fine.

        # For the example, let's use a force that can move the cart
        actual_force = random_action * env.force_mag
        # However, the env.step already clips the raw action if it's outside [-force_mag, force_mag]
        # So, if policy outputs -1 to 1, and force_mag is 10, then action needs to be scaled before passing to step,
        # or policy should output values in [-10, 10].
        # The current policy outputs tanh (-1,1). Let's assume this range is scaled by force_mag
        # by the agent or before passing to env.step.
        # For this example, let's just send a value that is within force_mag range.
        test_action = np.random.uniform(low=-env.force_mag, high=env.force_mag, size=(1,))


        print(f"Action: {test_action}")
        state, reward, done, _ = env.step(test_action)
        env.render() # Call render
        print(f"Step {i+1}: State={state[0]:.2f},{state[2]:.2f}, Reward={reward}, Done={done}")

        if done:
            print("Episode finished after {} steps.".format(i+1))
            # On done, reset the environment for a new episode if you want to continue
            # env.reset()
            break
    # if not done:
    # print("Episode ran for 50 steps.")

    env.close() # Close the visualizer
