import numpy as np
import matplotlib.pyplot as plt

def plot_results(results):
    """
    Plot metrics from experiment results
    
    Args:
        results: Dict of metrics for each agent variant
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    for agent_name, metrics in results.items():
        # Use moving average for smoother curve
        rewards = np.array(metrics['episode_rewards'])
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(moving_avg, label=agent_name)
    plt.title('Episode Rewards (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    for agent_name, metrics in results.items():
        eval_rewards = metrics['eval_rewards']
        eval_episodes = np.arange(0, len(metrics['episode_rewards']), 50)[:len(eval_rewards)]
        plt.plot(eval_episodes, eval_rewards, label=agent_name)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    for agent_name, metrics in results.items():
        # Use moving average for smoother curve
        lengths = np.array(metrics['episode_lengths'])
        moving_avg = np.convolve(lengths, np.ones(100)/100, mode='valid')
        plt.plot(moving_avg, label=agent_name)
    plt.title('Episode Lengths (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot win rates over time
    plt.subplot(2, 2, 4)
    window_size = 100
    for agent_name, metrics in results.items():
        rewards = np.array(metrics['episode_rewards'])
        win_rates = []
        for i in range(window_size, len(rewards), window_size):
            win_rate = np.sum(rewards[i-window_size:i] > 0) / window_size
            win_rates.append(win_rate)
        plt.plot(np.arange(window_size, len(rewards), window_size)[:len(win_rates)], 
                 win_rates, label=agent_name)
    plt.title('Win Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ppo_variants_comparison.png')
    plt.show()