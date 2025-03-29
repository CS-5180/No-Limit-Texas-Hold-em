import random
from typing import Dict, List, Any


class RandomAgent:
    """
    Agent that selects random actions from the legal action space.
    Useful for testing environment functionality and as a baseline.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the random agent.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.name = "Random Agent"
        self.episode_rewards = []

    def act(self, observation: Dict, legal_actions: List[int] = None) -> int:
        """
        Select a random legal action.

        Args:
            observation: Current game state observation
            legal_actions: List of legal action IDs (if None, use observation)

        Returns:
            Selected action ID
        """
        # Get legal actions from observation if not provided
        if legal_actions is None:
            legal_actions = list(observation.get('legal_actions', {}).keys())

        # Default to action 0 if no legal actions (should never happen)
        if not legal_actions:
            return 0

        # Choose random action from legal actions
        return self.rng.choice(legal_actions)

    def reset(self):
        """Reset the agent state for a new episode."""
        pass

    def observe(self, observation: Dict, action: int, reward: float, next_observation: Dict, done: bool):
        """
        Record an observation-action-reward-next_observation transition.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
        """
        # For random agent, we only track final rewards
        if done:
            self.episode_rewards.append(reward)

    def get_stats(self) -> Dict:
        """
        Get agent statistics.

        Returns:
            Dict of statistics
        """
        if not self.episode_rewards:
            return {'mean_reward': 0, 'num_episodes': 0}

        return {
            'mean_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'num_episodes': len(self.episode_rewards),
            'total_reward': sum(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'max_reward': max(self.episode_rewards)
        }