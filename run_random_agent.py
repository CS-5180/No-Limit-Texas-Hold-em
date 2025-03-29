import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rlcard.utils.utils import set_seed
from rlcard.utils.logger import Logger

# Import our environment wrapper and agent
from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.random_agent import RandomAgent


def run_episode(env, agent, render=False):
    """
    Run a single episode with the agent.

    Args:
        env: Environment to run in
        agent: Agent to use for action selection
        render: Whether to render the environment

    Returns:
        Tuple of (episode_reward, episode_length)
    """
    # Reset the environment and agent
    observation = env.reset()
    agent.reset()

    done = False
    episode_reward = 0
    step_count = 0

    # Run until episode terminates
    while not done:
        # Select action
        action = agent.act(observation)

        # Render if requested
        if render:
            env.render()
            print(f"Selected action: {env._action_to_string(action)}")
            time.sleep(1)  # Slow down rendering for human viewing

        # Take action in environment
        next_observation, reward, done, info = env.step(action)

        # Update agent with transition
        agent.observe(observation, action, reward, next_observation, done)

        # Update for next step
        observation = next_observation
        episode_reward += reward
        step_count += 1

    # Final render to show outcome
    if render:
        print(f"\nEpisode complete! Final reward: {episode_reward}")
        print(f"Episode rewards for all players: {info['episode_rewards']}")

    return episode_reward, step_count


def run_multiple_episodes(env, agent, num_episodes=100, render_interval=0):
    """
    Run multiple episodes and collect statistics.

    Args:
        env: Environment to run in
        agent: Agent to use
        num_episodes: Number of episodes to run
        render_interval: How often to render (0 = never)

    Returns:
        List of episode rewards
    """
    episode_rewards = []
    episode_lengths = []

    # Run for specified number of episodes
    for episode in tqdm(range(1, num_episodes + 1), desc="Episodes"):
        # Determine whether to render this episode
        render = (render_interval > 0 and episode % render_interval == 0)

        # Run the episode
        reward, length = run_episode(env, agent, render)

        # Record statistics
        episode_rewards.append(reward)
        episode_lengths.append(length)

        # Show progress
        if episode % 10 == 0:
            mean_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {episode}: Mean reward over last 10 episodes: {mean_reward:.2f}")

    return episode_rewards, episode_lengths


def visualize_results(episode_rewards, episode_lengths, window_size=10):
    """
    Visualize the results from multiple episodes.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        window_size: Size of moving average window
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.5, label='Episode Reward')

    # Calculate and plot moving average reward
    if len(episode_rewards) >= window_size:
        moving_avg = [np.mean(episode_rewards[i:i + window_size])
                      for i in range(len(episode_rewards) - window_size + 1)]
        ax1.plot(range(window_size - 1, len(episode_rewards)), moving_avg,
                 label=f'{window_size}-Episode Moving Average')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True)

    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.5, label='Episode Length')

    # Calculate and plot moving average length
    if len(episode_lengths) >= window_size:
        moving_avg = [np.mean(episode_lengths[i:i + window_size])
                      for i in range(len(episode_lengths) - window_size + 1)]
        ax2.plot(range(window_size - 1, len(episode_lengths)), moving_avg,
                 label=f'{window_size}-Episode Moving Average')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save figure
    plt.savefig('results/random_agent_performance.png')
    plt.show()


def main(episodes=100, render=0, seed=42):
    """
    Main entry point for running random agent experiments.

    Args:
        episodes: Number of episodes to run
        render: Render every N episodes (0 = never)
        seed: Random seed for reproducibility
    """
    # Set random seeds
    np.random.seed(seed)
    set_seed(seed)

    # Create environment and agent
    env = RLCardPokerEnv(num_players=2)
    agent = RandomAgent(seed=seed)

    print(f"Running {episodes} episodes with random agent...")

    # Create RLCard logger for nice summary statistics
    logger = Logger(episodes)

    # Run episodes
    episode_rewards, episode_lengths = run_multiple_episodes(
        env, agent, episodes, render
    )

    # Display agent statistics
    stats = agent.get_stats()
    print("\nAgent Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Visualize results
    visualize_results(episode_rewards, episode_lengths)

    # Clean up
    env.close()

if __name__ == "__main__":
    # Call main function with default parameters
    # You can modify these parameters directly here
    main(episodes=100, render=10, seed=42)
