import numpy as np
import torch
from tqdm import tqdm
import copy

from utils.evaluation import evaluate_agent
from utils.poker_utils import shaped_reward
from agents.random_agent import RandomAgent

def train_agent(agent, env, episodes=1000, max_steps=200, 
              update_frequency=128, eval_frequency=50, 
              render_eval=False, seed=42, eval_episodes=10,
              use_shaped_rewards=True, use_entropy_decay=True,
              save_dir='checkpoints', save_best=True, opponents=None, snapshot_interval=500):
    """
    Train an agent in the given environment
    
    Args:
        agent: Agent to train
        env: Environment to train in
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        update_frequency: Frequency of policy updates
        eval_frequency: Frequency of evaluation
        render_eval: Whether to render during evaluation
        seed: Random seed
        eval_episodes: determines how many episodes will be run when evaluating agent's performance
        use_shaped_rewards: Whether to use shaped rewards for poker
        use_entropy_decay: Whether to apply entropy decay during training
        save_dir: Directory to save checkpoints
        save_best: Whether to save the best model based on evaluation
        opponents: List of opponent agents to sample from
        snapshot_interval: Frequency (in episodes) to snapshot agent into opponent pool
        
    Returns:
        Dict of metrics
    """

    # Print hyperparameters for verification
    print("\n===== Training Configuration =====")
    print(f"Agent Type: {agent.__class__.__name__}")
    print(f"Episodes: {episodes}")
    print(f"Max Steps: {max_steps}")
    print(f"Update Frequency: {update_frequency}")
    print(f"Evaluation Frequency: {eval_frequency}")
    print(f"Evaluation Episodes: {eval_episodes}")
    print(f"Random Seed: {seed}")
    print(f"Using Shaped Rewards: {use_shaped_rewards}")
    print(f"Using Entropy Decay: {use_entropy_decay}")

    # Print agent hyperparameters
    print("\n===== Agent Configuration =====")
    print(f"Learning Rate: {agent.lr}")
    print(f"Discount Factor (gamma): {agent.gamma}")
    print(f"GAE Lambda: {agent.gae_lambda}")
    print(f"Value Coefficient: {agent.value_coef}")
    print(f"Entropy Coefficient: {agent.entropy_coef}")
    print(f"Max Gradient Norm: {agent.max_grad_norm}")
    print(f"Update Epochs: {agent.update_epochs}")
    print(f"Minibatch Size: {agent.minibatch_size}")
    print(f"Hidden Dimension: {agent.hidden_dim}")
    print(f"Constraint Type: {agent.constraint_type}")

    if agent.constraint_type == 'clip':
        print(f"Clip Ratio: {agent.clip_ratio}")
    elif agent.constraint_type == 'kl':
        print(f"KL Target: {agent.kl_target}")
        print(f"Initial KL Coefficient: {agent.kl_coef if hasattr(agent, 'kl_coef') else 'N/A'}")


    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    wins = 0
    step_count = 0

    # Best model tracking
    best_eval_reward = float('-inf')
    best_win_rate = 0.0

    # Create checkpoint directory
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Entropy decay setup
    if use_entropy_decay and hasattr(agent, 'entropy_coef'):
        initial_entropy = agent.entropy_coef
        min_entropy = 0.01
        decay_rate = 0.999995  # Very slow decay

    milestone_episodes = [0, 1000, 5000, 10000, 15000, 20000]
    
    # Training loop
    for episode in tqdm(range(episodes), desc=f"Training {agent.__class__.__name__}"):
        # === Opponent Sampling ===
        if opponents:
            opponent = np.random.choice(opponents)
            env.set_agents([agent, opponent])
        else:
            env.set_agents([agent, RandomAgent()])

        # === Self-play Opponent Snapshotting ===
        if opponents is not None and episode % snapshot_interval == 0 and episode != 0:
            opponents.append(copy.deepcopy(agent))

        observation = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # Apply entropy decay
        if use_entropy_decay and hasattr(agent, 'entropy_coef'):
            agent.entropy_coef = max(initial_entropy * (decay_rate ** episode), min_entropy)
            if episode % 100 == 0:
                print(f"Current entropy coefficient: {agent.entropy_coef:.5f}")
        
        while not done and step < max_steps:
            # Extract state and legal actions
            state = observation['observation']
            legal_actions = observation['legal_actions']
            
            # Select action
            action, log_prob, value = agent.select_action(state, legal_actions)
            
            # Take action in environment
            next_observation, reward, done, info = env.step(action)

            # Apply reward shaping if enabled
            if use_shaped_rewards:
                modified_reward = shaped_reward(observation, action, reward, next_observation, done, info)
            else:
                modified_reward = reward
            
            # Store transition
            agent.store_transition(
                state, action, modified_reward, next_observation['observation'],
                done, log_prob, value
            )

            # Update observation
            observation = next_observation
            episode_reward += reward
            step += 1
            step_count += 1
            
            # Update policy if enough data is collected
            if step_count % update_frequency == 0:
                agent.learn()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Win tracking
        if episode_reward > 0:
            wins += 1
        
        # Evaluate periodically
        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, env, episodes=eval_episodes, render=render_eval)
            eval_rewards.append(eval_reward)

            # Calculate current win rate
            current_win_rate = wins / (episode + 1) if episode > 0 else 0

            # Print progress
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, "
                  f"Win Rate: {current_win_rate:.2f}, Eval Reward: {eval_reward:.2f}")

            # Save checkpoint only at milestone episodes
            if episode in milestone_episodes:
                checkpoint_path = f"{save_dir}/{agent.__class__.__name__}_episode_{episode}.pt"
                agent.save_model(checkpoint_path)
                print(f"Milestone checkpoint saved at episode {episode}")

            # Save best model based on evaluation reward
            if save_best and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_path = f"{save_dir}/{agent.__class__.__name__}_best_eval.pt"
                agent.save_model(best_model_path)
                print(f"New best model saved with eval reward: {best_eval_reward:.2f}")

            # Also save best model based on win rate (alternative metric)
            if save_best and current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                best_winrate_path = f"{save_dir}/{agent.__class__.__name__}_best_winrate.pt"
                agent.save_model(best_winrate_path)
                print(f"New best model saved with win rate: {best_win_rate:.2f}")

    # Save final model
    final_path = f"{save_dir}/{agent.__class__.__name__}_final.pt"
    agent.save_model(final_path)

    # Gather metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'win_rate': wins / episodes,
        'agent_type': agent.__class__.__name__,
        'best_eval_reward': best_eval_reward,
        'best_win_rate': best_win_rate
    }

    return metrics