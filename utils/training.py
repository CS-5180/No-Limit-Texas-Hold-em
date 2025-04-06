import numpy as np
import torch
from tqdm import tqdm
import time

from utils.evaluation import evaluate_agent

def train_agent(agent, env, episodes=1000, max_steps=200, 
              update_frequency=128, eval_frequency=50, 
              render_eval=False, seed=42, eval_episodes=10):
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
        
    Returns:
        Dict of metrics
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    wins = 0
    step_count = 0
    
    # Training loop
    for episode in tqdm(range(episodes), desc=f"Training {agent.__class__.__name__}"):
        observation = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Extract state and legal actions
            state = observation['observation']
            legal_actions = observation['legal_actions']
            
            # Select action
            action, log_prob, value = agent.select_action(state, legal_actions)
            
            # Take action in environment
            next_observation, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(
                state, action, reward, next_observation['observation'], 
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
            
            # Print progress
            win_rate = wins / (episode + 1)
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, "
                  f"Win Rate: {win_rate:.2f}, Eval Reward: {eval_reward:.2f}")
    
    # Gather metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'win_rate': wins / episodes,
        'agent_type': agent.__class__.__name__
    }
    
    return metrics