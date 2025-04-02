import os
import numpy as np
import torch

from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.ppo_clip import PPO_CLIP
from agents.ppo_kl import PPO_KL
from utils.training import train_agent
from experiments.visualization import plot_results

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
    
    # Common agent parameters
    agent_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 128,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'update_epochs': 4,
        'minibatch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Training parameters
    train_params = {
        'episodes': episodes,
        'max_steps': 200,
        'update_frequency': 128,
        'eval_frequency': 50,
        'render_eval': False
    }
    
    # Results storage
    results = {}
    trained_agents = {}
    
    # Run experiments for each agent variant
    for agent_class in agent_classes:
        print(f"\nTraining {agent_class.__name__}")
        
        # Initialize agent
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
        print(f"  Win rate: {metrics['win_rate']:.4f}")
    
    return results, trained_agents