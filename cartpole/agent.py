import torch
import torch.optim as optim
from cartpole.policy import PolicyNetwork
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma  # Discount factor

        # These lists will store rewards and log_probs for one episode
        self.rewards = []
        self.log_probs = []

    def select_action(self, state, deterministic=False):
        # The policy's select_action method returns action and log_prob
        # action is already a suitable format for the environment (e.g., numpy array)
        action, log_prob = self.policy.select_action(state, deterministic=deterministic)

        # Store log_prob for the update, only if not deterministic
        if not deterministic:
            self.log_probs.append(log_prob)

        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        if not self.log_probs: # No log_probs typically means deterministic mode or no steps taken
            if self.rewards: # Clear rewards if any, for consistency
                del self.rewards[:]
            return

        # Calculate discounted returns (Gt)
        R = 0
        returns = []
        # Iterate backwards through self.rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R) # Prepend R to maintain order

        # Convert returns to a PyTorch tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability:
        # (returns - returns.mean()) / (returns.std() + eps)
        # Ensure there's more than one return value before calculating std
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # If len(returns) == 1, 'returns' contains a single element [G0].
        # We don't normalize it to 0, allowing single-step episodes to contribute to learning
        # if G0 itself is not 0. The previous 'else' branch made it (G0-G0)/eps = 0.

        # Calculate policy loss
        policy_loss = []
        # Iterate through self.log_probs and calculated returns
        for log_prob, R_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R_t)

        # Sum up all losses for the episode
        # torch.stack can be used if log_probs are 0-dim tensors (scalars)
        # If log_probs are already tensors, ensure they are compatible for cat or stack
        # Assuming log_prob from policy.select_action is a scalar tensor or can be made so
        if not policy_loss: # Should not happen if log_probs is not empty
            del self.rewards[:]
            del self.log_probs[:]
            return

        self.optimizer.zero_grad()
        # Concatenate or stack depends on the shape of elements in policy_loss
        # If each -log_prob * R_t is a scalar tensor, stack them then sum.
        policy_loss_tensor = torch.stack(policy_loss).sum()

        policy_loss_tensor.backward()
        self.optimizer.step()

        # Clear stored rewards and log probabilities for the next episode
        del self.rewards[:]
        del self.log_probs[:]

if __name__ == '__main__':
    # Example Usage
    state_dim = 4
    action_dim = 1
    agent = Agent(state_dim, action_dim)

    # Simulate a few steps in an episode
    example_state = np.random.rand(state_dim)

    # Step 1
    action1 = agent.select_action(example_state)
    # Assume action1 results in reward1 from environment
    reward1 = 1.0
    agent.store_reward(reward1)
    print(f"State: {example_state}, Action: {action1}, Reward: {reward1}, LogProb: {agent.log_probs[-1]}")

    # Step 2
    next_state = np.random.rand(state_dim) # New state from environment
    action2 = agent.select_action(next_state)
    reward2 = 0.8 # Assume pole is more tilted
    agent.store_reward(reward2)
    print(f"State: {next_state}, Action: {action2}, Reward: {reward2}, LogProb: {agent.log_probs[-1]}")

    # Step 3 - Terminal state
    final_state = np.random.rand(state_dim)
    action3 = agent.select_action(final_state)
    reward3 = -0.5 # Or whatever the terminal reward is based on cos(theta)
    agent.store_reward(reward3) # Store final reward
    print(f"State: {final_state}, Action: {action3}, Reward: {reward3}, LogProb: {agent.log_probs[-1]}")

    print(f"\nRewards stored: {agent.rewards}")
    print(f"Log_probs stored: {agent.log_probs}")

    # Episode ends, update policy
    agent.update_policy()

    print("\nAfter policy update:")
    print(f"Rewards stored: {agent.rewards} (should be empty)")
    print(f"Log_probs stored: {agent.log_probs} (should be empty)")

    # Test with deterministic action
    print("\nTesting deterministic action selection:")
    action_det = agent.select_action(example_state, deterministic=True)
    print(f"Deterministic action: {action_det}")
    print(f"Log_probs stored: {agent.log_probs} (should be empty as it was deterministic)")
    # store_reward and update_policy would typically not be called after deterministic actions
    # unless it's part of a specific evaluation protocol that still involves learning.
    # For REINFORCE, learning happens based on stochastic actions.

    # Test update with single step episode
    print("\nTesting update with single step episode:")
    agent.select_action(example_state) # log_prob is stored
    agent.store_reward(5.0)
    print(f"Rewards: {agent.rewards}, LogProbs: {len(agent.log_probs)}")
    agent.update_policy()
    print(f"Rewards: {agent.rewards}, LogProbs: {len(agent.log_probs)} (should be empty)")
    print("Single step update completed.")

    print("\nAgent initialized and example episode processed.")
    # Check if policy parameters changed (requires more detailed check)
    # initial_params = [p.clone() for p in agent.policy.parameters()]
    # ... run update ...
    # updated_params = [p for p in agent.policy.parameters()]
    # for i_param, u_param in zip(initial_params, updated_params):
    #     if not torch.equal(i_param, u_param):
    #         print("Policy parameters changed after update.")
    #         break
    # else:
    #     print("Policy parameters did NOT change after update (check logic if this is unexpected).")
