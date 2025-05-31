# cartpole/main.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from cartpole.cartpole_env import CartPoleEnv
from cartpole.agent import Agent

def main():
    # Hyperparameters
    state_dim = 4  # For CartPole: [cart_pos, cart_vel, pole_angle, pole_vel]
    action_dim = 1 # For CartPole: [force]
    lr = 1e-3
    gamma = 0.99
    num_episodes = 1000 # Number of episodes to train for
    max_steps_per_episode = 500 # Max steps before an episode is truncated
    render_every_n_episodes = 100 # Render one episode every N training episodes
    log_interval = 20 # Print average reward every N episodes

    env = CartPoleEnv()
    # Note: In a more general setting, state_dim and action_dim would be derived from
    # env.observation_space.shape[0] and env.action_space.shape[0] (for Box spaces)
    # or other properties of env.action_space (e.g., .n for Discrete spaces).
    # Our custom CartPoleEnv doesn't formally define these Gym spaces yet.

    agent = Agent(state_dim, action_dim, lr=lr, gamma=gamma)

    episode_rewards = []
    average_rewards_over_time = [] # To plot average rewards

    print(f"Starting training for {num_episodes} episodes...")
    print(f"Hyperparameters: lr={lr}, gamma={gamma}, max_steps_per_episode={max_steps_per_episode}")
    print(f"Logging average reward every {log_interval} episodes.")
    print(f"Rendering an episode every {render_every_n_episodes} episodes.")

    for episode_num in range(1, num_episodes + 1):
        state = env.reset()
        current_episode_reward = 0.0

        # Determine if this episode should be rendered
        # Render if it's an interval episode OR the very last episode
        should_render_this_episode = (episode_num % render_every_n_episodes == 0) or \
                                     (episode_num == num_episodes)

        if should_render_this_episode:
            print(f"Rendering episode {episode_num}...")

        for step_num in range(max_steps_per_episode):
            if should_render_this_episode:
                env.render(mode='human')

            # Agent selects an action. Policy output is scaled by force_mag inside agent or env.
            # Our current PolicyNetwork outputs tanh (-1,1). Agent passes this to env.
            # Env's step() clips action to [-force_mag, force_mag] and uses action[0].
            action_from_policy = agent.select_action(state, deterministic=False)

            # Scale action from policy (tanh output: -1 to 1) to environment's force range
            # This step is crucial if the policy network's output range (e.g. tanh gives [-1,1])
            # is different from the force magnitude the environment expects.
            # The environment's step function currently clips any action value using self.force_mag,
            # and its physics expects the action to be the actual force.
            # So, if policy outputs [-1, 1], we should scale it by self.force_mag.
            # However, our agent.select_action already returns a value that is NOT yet scaled by force_mag.
            # The policy's mean is tanh. The CartPoleEnv.step() takes action and clips it.
            # Let's assume for now that the agent's action output is already in the correct scale for the environment,
            # or that the environment's clipping is sufficient if the policy outputs values like [-10, 10].
            # The current PolicyNetwork outputs tanh(mean) which is [-1,1].
            # The CartPoleEnv.step uses action[0] and clips it to [-self.force_mag, self.force_mag].
            # So, if policy outputs e.g. 0.5, and force_mag is 10, env will use 0.5. This is probably too small.
            # It's better to scale the action from policy before passing to env.
            # action_to_env = action_from_policy * env.force_mag # Scale action
            # Then ensure action_to_env is np.array([value]) if env.step expects that.
            # PolicyNetwork returns np.array([value]), so action_from_policy * env.force_mag is fine.

            # Given PolicyNetwork outputs mean in [-1,1] (due to tanh) and CartPoleEnv has force_mag=10.0
            # It is conventional for the agent to scale its action.
            scaled_action = action_from_policy * env.force_mag

            next_state, reward, done, _ = env.step(scaled_action)

            agent.store_reward(reward)
            current_episode_reward += reward
            state = next_state

            if done:
                break

        agent.update_policy()

        episode_rewards.append(current_episode_reward)

        if episode_num % log_interval == 0:
            # Calculate average of the last 'log_interval' rewards
            avg_reward = np.mean(episode_rewards[-log_interval:])
            average_rewards_over_time.append(avg_reward)
            print(f"Episode {episode_num}/{num_episodes} | Steps: {step_num+1} | Total Reward: {current_episode_reward:.2f} | Average Reward (last {log_interval}): {avg_reward:.2f}")

        if should_render_this_episode and env.visualizer:
            print(f"Finished episode {episode_num}. Visualization window will remain or close on next env.close().")
            # To close window immediately after render:
            # env.close() # This would close and reopen visualizer each time. Not ideal.
            # The current setup keeps the window if visualizer exists, and main env.close() shuts it.

    env.close()
    print("Training finished.")

    # Plotting results
    plt.figure(figsize=(12, 7))
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.7)

    # Calculate x-coordinates for average rewards plot to align with episode numbers
    # Each point in average_rewards_over_time corresponds to the end of a log_interval
    avg_reward_x_coords = [(i + 1) * log_interval for i in range(len(average_rewards_over_time))]
    plt.plot(avg_reward_x_coords, average_rewards_over_time, label=f'Average Reward (every {log_interval} episodes)', linewidth=2.5, color='red', marker='o', markersize=5)

    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward per Episode")
    plt.title(f"REINFORCE Training on Custom CartPole\n(lr={lr}, gamma={gamma}, max_steps={max_steps_per_episode})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    plot_filename = "cartpole_training_rewards.png"
    try:
        plt.savefig(plot_filename)
        print(f"Saved training rewards plot to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # plt.show() # Uncomment if you want to display the plot interactively after training

if __name__ == '__main__':
    main()
```
