import numpy as np
import time

def evaluate_agent(agent, env, episodes=10, render=False):
    """
    Evaluate an agent's performance without training
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        Average reward over evaluation episodes
    """
    rewards = []
    
    for _ in range(episodes):
        observation = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Extract state and legal actions
            state = observation['observation']
            legal_actions = observation['legal_actions']
            
            # Select action (deterministic)
            action, _, _ = agent.select_action(state, legal_actions)
            
            # Take action in environment
            next_observation, reward, done, info = env.step(action)
            
            if render:
                env.render()
                time.sleep(0.1)  # Slow down rendering
            
            # Update state and reward
            observation = next_observation
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)