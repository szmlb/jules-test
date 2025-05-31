import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc_mean = nn.Linear(128, action_dim)
        self.fc_log_std = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # Output mean typically bounded, e.g. by tanh, if actions are bounded.
        # For CartPole, the force can be positive or negative.
        # Let's assume the environment or agent will scale this mean appropriately.
        # A common range for tanh is [-1, 1]. If force_mag is 10,
        # then action = mean * force_mag.
        mean = torch.tanh(self.fc_mean(x))

        # log_std is output directly. Add a small constant for stability if needed,
        # or clip log_std to a certain range.
        # For example: log_std = torch.clamp(self.fc_log_std(x), min=-20, max=2)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)  # Ensure std is positive
        return mean, std

    def select_action(self, state, deterministic=False):
        # Convert state to a PyTorch tensor if it's not already.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Add a batch dimension if processing a single state and it doesn't have one.
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mean, std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = None # No stochasticity, so log_prob is not usually computed
        else:
            distribution = Normal(mean, std)
            action = distribution.sample()
            # Calculate the log probability of the action.
            # sum(dim=-1) is important if action_dim > 1, for 1D action it's just log_prob(action)
            log_prob = distribution.log_prob(action).sum(dim=-1)

        # If state had a batch dim added, remove it for the action returned to env
        if state.shape[0] == 1 and action.shape[0] == 1 : # if a single state was unsqueezed
             # For CartPole, action is 1D. We want to return a scalar-like value or 1D array for the env.
             # action.squeeze(0) would give a 1D tensor if action_dim > 1, or 0D tensor if action_dim == 1.
             # .item() is for scalar tensors. .cpu().detach().numpy() is more general.
            if action.numel() == 1: # If action is a single value
                action_val = action.squeeze(0).item() # Return as a Python scalar
            else:
                action_val = action.squeeze(0).cpu().detach().numpy() # Return as NumPy array
        else: # Batch of actions
            action_val = action.cpu().detach().numpy()


        # For CartPole (action_dim=1), action.item() is fine if it's a single action.
        # If batch processing, return the tensor or numpy array.
        # Assuming select_action is called with one state at a time during rollout.
        # The environment expects a single force value (or a list/array with one value).
        # action.item() works if action is a single-element tensor.
        # For a batch_size=1, action will be shape [1, action_dim].
        # We need to return something like `[force_value]` or `force_value`
        # The current env.step expects action to be clippable like `action[0]`
        # So, action_val should be `[value]` or `np.array([value])`

        # Let's ensure output is a numpy array for consistency, as env step uses action[0]
        if isinstance(action_val, float):
            action_val = np.array([action_val], dtype=np.float32)

        return action_val, log_prob


if __name__ == '__main__':
    # Example Usage
    state_dim = 4  # Cart position, cart velocity, pole angle, pole angular velocity
    action_dim = 1 # Force applied to the cart

    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim)

    # Example state (random)
    example_state = np.random.rand(state_dim).astype(np.float32)
    print(f"Example state: {example_state}")

    # Select an action
    action, log_prob = policy.select_action(example_state)
    print(f"Selected action: {action} (Type: {type(action)})")
    print(f"Log probability of the action: {log_prob}")

    # Select a deterministic action
    deterministic_action, _ = policy.select_action(example_state, deterministic=True)
    print(f"Selected deterministic action: {deterministic_action} (Type: {type(deterministic_action)})")

    # Example batch of states
    batch_size = 3
    example_batch_state = np.random.rand(batch_size, state_dim).astype(np.float32)
    print(f"\nExample batch state (shape): {example_batch_state.shape}")

    # Process batch (forward pass)
    # Convert batch state to tensor
    example_batch_state_tensor = torch.tensor(example_batch_state, dtype=torch.float32)
    means, stds = policy(example_batch_state_tensor)
    print(f"Means (shape): {means.shape}")
    print(f"Stds (shape): {stds.shape}")

    # Select actions for a batch (though select_action is designed for single state)
    # To do this properly, select_action would need to handle batch inputs more directly for action return
    # The current select_action is more for single instance -> single action value for env
    # For training, one might call forward() directly and then sample.

    # Test the output type of select_action for the environment
    # env.step(action) where action is from select_action.
    # The CartPoleEnv.step expects `action` to be such that `action[0]` works.
    # So, `np.array([value])` is a good format.
    if isinstance(action, np.ndarray) and action.ndim == 1 and action.shape[0] == action_dim:
        print("\nAction format appears compatible with env.step's use of action[0].")
    else:
        print(f"\nAction format might be incompatible: {action}, type {type(action)}")

    # Check if torch is using GPU
    if torch.cuda.is_available():
        print("\nPyTorch is using GPU.")
        device = torch.device('cuda')
        policy.to(device)
        example_state_gpu = torch.tensor(example_state, dtype=torch.float32).to(device)
        action_gpu, log_prob_gpu = policy.select_action(example_state_gpu)
        print(f"Selected action (GPU): {action_gpu}")
        print(f"Log probability (GPU): {log_prob_gpu}")
    else:
        print("\nPyTorch is using CPU.")
