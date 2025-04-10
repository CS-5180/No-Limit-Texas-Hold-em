import os
import numpy as np
import torch

# Import configurations
from configs.ppo_clip_config import PPO_CLIP_CONFIG
from configs.ppo_kl_config import PPO_KL_CONFIG
from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.ppo_clip import PPO_CLIP
from agents.ppo_kl import PPO_KL
from utils.training import train_agent
from experiments.visualization import plot_results
from agents.random_agent import RandomAgent

def run_ablation_study(episodes=1000, save_dir='models'):
    """
    Run ablation study comparing PPO variants on poker environment
    
    Args:
        episodes: Number of episodes to train each agent
        save_dir: Directory to save trained models
        
    Returns:
        Dict of results and trained agents
    """
    # Create environment
    env = RLCardPokerEnv(num_players=2)
    
    # Get state and action dimensions
    observation = env.reset()
    state_dim = len(observation['observation'])
    action_dim = env.action_space['n']
    
    # Define agents to test
    agent_classes = [
        PPO_CLIP,  # PPO with clipped objective
        PPO_KL     # PPO with KL constraint
    ]

    # Initialize opponent pool with at least one RandomAgent
    opponents = [RandomAgent()]
    
    # Common agent parameters
    common_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 512,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,

        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.05,
        'max_grad_norm': 0.5,

        'update_epochs': 10,
        'minibatch_size': 128,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        #'entropy_decay': 0.9999,
    }

    # Training parameters
    train_params = {
        'episodes': episodes,
        'max_steps': 200,
        'update_frequency': 256,
        'eval_frequency': 500,
        'render_eval': False,
        'eval_episodes': 50,
        'use_shaped_rewards': False,
        'use_entropy_decay': False,
        'opponents': opponents,
        'snapshot_interval': 500
    }
    
    # Results storage
    results = {}
    trained_agents = {}

    print(f"Running on device {common_params['device']}")
    
    # Run experiments for each agent variant
    for agent_class in agent_classes:
        print(f"\nTraining {agent_class.__name__}")

        # Select the appropriate config based on agent type
        if agent_class == PPO_CLIP:
            # Merge common parameters with CLIP-specific parameters
            agent_params = {**common_params, **PPO_CLIP_CONFIG}
        else:  # PPO_KL
            # Merge common parameters with KL-specific parameters
            agent_params = {**common_params, **PPO_KL_CONFIG}

        # Initialize agent with appropriate parameters
        agent = agent_class(**agent_params)
        
        # Train agent
        metrics = train_agent(agent, env, **train_params)
        
        # Store results
        results[agent_class.__name__] = metrics
        trained_agents[agent_class.__name__] = agent
        
        # Save model
        os.makedirs(save_dir, exist_ok=True)
        agent.save_model(f"{save_dir}/{agent_class.__name__}.pt")

    # Plot results
    plot_results(results)

    # Print final metrics
    print("\n--- Final Results ---")
    for agent_name, metrics in results.items():
        print(f"{agent_name}:")
        print(f"  Average reward (last 100 episodes): {np.mean(metrics['episode_rewards'][-100:]):.4f}")
        print(f"  Final win rate: {metrics['win_rate']:.4f}")

        # Add best metrics from checkpoints
        if 'best_eval_reward' in metrics:
            print(f"  Best evaluation reward: {metrics['best_eval_reward']:.4f}")
        if 'best_win_rate' in metrics:
            print(f"  Best win rate: {metrics['best_win_rate']:.4f}")
    
    return results, trained_agents