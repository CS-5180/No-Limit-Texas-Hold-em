import os
import argparse
import torch
import numpy as np

from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.ppo_clip import PPO_CLIP
from agents.ppo_kl import PPO_KL
from experiments.ablation_study import run_ablation_study
from configs.default_config import (
    ENV_CONFIG, 
    AGENT_CONFIG, 
    TRAINING_CONFIG
)
from configs.ppo_clip_config import PPO_CLIP_CONFIG
from configs.ppo_kl_config import PPO_KL_CONFIG

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate PPO agents on Poker')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='ablation', 
                        choices=['ablation', 'train', 'eval', 'plot'],
                        help='Operation mode')
    
    # Agent arguments
    parser.add_argument('--agent', type=str, default='ppo_clip',
                        choices=['ppo_clip', 'ppo_kl'],
                        help='Agent type to train/evaluate')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=TRAINING_CONFIG['episodes'],
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=AGENT_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['seed'],
                        help='Random seed')
    
    # Evaluation arguments
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    
    return parser.parse_args()

def train_single_agent(args):
    """Train a single agent"""
    # Create environment
    env = RLCardPokerEnv(**ENV_CONFIG)
    
    # Get state and action dimensions
    observation = env.reset()
    state_dim = len(observation['observation'])
    action_dim = env.action_space['n']
    
    # Select agent type and config
    if args.agent == 'ppo_clip':
        agent_class = PPO_CLIP
        agent_config = PPO_CLIP_CONFIG
    else:  # ppo_kl
        agent_class = PPO_KL
        agent_config = PPO_KL_CONFIG
    
    # Override config with args
    agent_config['lr'] = args.lr
    agent_config['device'] = 'mps' if torch.mps.is_available() else 'cpu'
    
    # Initialize agent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config
    )
    
    # # Train agent
    # from utils.training import train_agent
    
    # training_config = TRAINING_CONFIG.copy()
    # training_config['episodes'] = args.episodes
    # training_config['seed'] = args.seed
    
    # metrics = train_agent(agent, env, **training_config)
    
    # # Save model
    # os.makedirs(args.save_dir, exist_ok=True)
    # agent.save_model(f"{args.save_dir}/{args.agent}_e{args.episodes}_s{args.seed}.pt")
    
    # return agent, metrics

    from utils.training import train_agent
    
    training_config = TRAINING_CONFIG.copy()
    training_config['episodes'] = args.episodes
    training_config['seed'] = args.seed
    
    metrics = train_agent(agent, env, **training_config)
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = f"{args.save_dir}/{args.agent}_e{args.episodes}_s{args.seed}.pt"
    agent.save_model(model_path)
    
    # Create a dictionary with this agent's results for the visualization function
    results = {args.agent: metrics}
    
    # Use the existing plot_results function from experiments.visualization
    from experiments.visualization import plot_results
    plot_results(results)
    
    # Optional: save metrics for later analysis
    np.save(f"logs/{args.agent}_metrics.npy", metrics)
    
    return agent, metrics

def evaluate_agent_from_file(args):
    """Evaluate a trained agent from file"""
    # Create environment
    env = RLCardPokerEnv(**ENV_CONFIG)
    
    # Get state and action dimensions
    observation = env.reset()
    state_dim = len(observation['observation'])
    action_dim = env.action_space['n']
    
    # Select agent type and config
    if args.agent == 'ppo_clip':
        agent_class = PPO_CLIP
        agent_config = PPO_CLIP_CONFIG
    else:  # ppo_kl
        agent_class = PPO_KL
        agent_config = PPO_KL_CONFIG
    
    # Initialize agent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_config
    )
    
    # Load model
    agent.load_model(args.model_path)
    
    # Evaluate agent
    from utils.evaluation import evaluate_agent
    
    reward = evaluate_agent(agent, env, episodes=50, render=args.render)
    print(f"Average reward over 50 episodes: {reward:.4f}")

def main():
    """Main function"""
    args = parse_args()
    
    if args.mode == 'ablation':
        # Run ablation study
        run_ablation_study(episodes=args.episodes, save_dir=args.save_dir)
    
    elif args.mode == 'train':
        # Train a single agent
        train_single_agent(args)
    
    elif args.mode == 'eval':
        # Evaluate an agent
        if args.model_path is None:
            print("Error: --model-path is required for evaluation")
            return
        
        evaluate_agent_from_file(args)
    
    elif args.mode == 'plot':
        # Plot metrics from a previous run
        import numpy as np
        from experiments.visualization import plot_results
        
        try:
            metrics = np.load(f"logs/{args.agent}_metrics.npy", allow_pickle=True).item()
            results = {args.agent: metrics}
            plot_results(results)
        except FileNotFoundError:
            print(f"No metrics file found for agent {args.agent}")

if __name__ == "__main__":
    main()