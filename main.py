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
                        choices=['ablation', 'train', 'eval', 'plot', 'checkpoint-ablation'],
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
    parser.add_argument('--compare-model-path', type=str, default=None,
                    help='Path to second model for comparison in ablation study')
    parser.add_argument('--eval-episodes', type=int, default=100,
                    help='Number of episodes for evaluation')
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

def ablation_from_checkpoints(args):
    """Perform ablation study using pre-trained checkpoints"""
    if args.model_path is None or args.compare_model_path is None:
        print("Error: Both --model-path and --compare-model-path are required")
        return
        
    # Create environment
    env = RLCardPokerEnv(**ENV_CONFIG)
    
    # Get state and action dimensions
    observation = env.reset()
    state_dim = len(observation['observation'])
    action_dim = env.action_space['n']
    
    # Determine agent types from file paths
    first_agent_type = 'ppo_clip' if 'clip' in args.model_path.lower() else 'ppo_kl'
    second_agent_type = 'ppo_clip' if 'clip' in args.compare_model_path.lower() else 'ppo_kl'
    
    # Initialize agents
    device = 'mps' if torch.mps.is_available() else 'cpu'
    
    # Initialize and load first agent
    if first_agent_type == 'ppo_clip':
        first_agent = PPO_CLIP(state_dim=state_dim, action_dim=action_dim, 
                         **{**PPO_CLIP_CONFIG, 'device': device})
    else:
        first_agent = PPO_KL(state_dim=state_dim, action_dim=action_dim,
                        **{**PPO_KL_CONFIG, 'device': device})
    first_agent.load_model(args.model_path)
    
    # Initialize and load second agent
    if second_agent_type == 'ppo_clip':
        second_agent = PPO_CLIP(state_dim=state_dim, action_dim=action_dim,
                          **{**PPO_CLIP_CONFIG, 'device': device})
    else:
        second_agent = PPO_KL(state_dim=state_dim, action_dim=action_dim,
                        **{**PPO_KL_CONFIG, 'device': device})
    second_agent.load_model(args.compare_model_path)
    
    # Evaluate agents
    from utils.evaluation import evaluate_agent
    
    print(f"Evaluating {first_agent_type.upper()} from {args.model_path}...")
    first_reward = evaluate_agent(first_agent, env, episodes=args.eval_episodes, 
                                 render=args.render)
    
    print(f"Evaluating {second_agent_type.upper()} from {args.compare_model_path}...")
    second_reward = evaluate_agent(second_agent, env, episodes=args.eval_episodes,
                                  render=args.render)
    
    # Record metrics
    results = {
        first_agent_type.upper(): {'eval_reward': first_reward},
        second_agent_type.upper(): {'eval_reward': second_reward}
    }
    
    # Print comparison
    print("\n=== Ablation Study Results ===")
    print(f"{first_agent_type.upper()} avg reward: {first_reward:.2f}")
    print(f"{second_agent_type.upper()} avg reward: {second_reward:.2f}")
    print(f"Difference: {first_reward - second_reward:.2f}")
    
    # Create bar chart for visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar([first_agent_type.upper(), second_agent_type.upper()], 
            [first_reward, second_reward])
    plt.title('PPO Variant Comparison')
    plt.ylabel('Average Reward')
    plt.savefig('ablation_results.png')
    plt.show()
    
    return results

def main():
    """Main function"""
    args = parse_args()
    
    if args.mode == 'ablation':
        # Run ablation study
        print("Running ablation study")
        run_ablation_study(episodes=args.episodes, save_dir=args.save_dir)
    
    elif args.mode == 'checkpoint-ablation':
        # Run ablation study with pre-trained checkpoints
        print("Running ablation study from checkpoints")
        ablation_from_checkpoints(args)

    elif args.mode == 'train':
        # Train a single agent
        print("Running training agent")
        train_single_agent(args)
    
    elif args.mode == 'eval':
        # Evaluate an agent
        print("Running evaluation agent")
        if args.model_path is None:
            print("Error: --model-path is required for evaluation")
            return
        
        evaluate_agent_from_file(args)
    
    elif args.mode == 'plot':
        print("Running plotting results")
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