from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        """
        Initialize the base agent
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize metrics
        self.episode_rewards = []
        self.losses = {}
    
    @abstractmethod
    def select_action(self, state, legal_actions):
        """
        Select an action given the current state
        
        Args:
            state: Current state observation
            legal_actions: List of legal actions
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store a transition in the agent's memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate
        """
        pass
    
    @abstractmethod
    def learn(self):
        """Learn from collected experiences"""
        pass
    
    def save_model(self, path):
        """
        Save model to path
        
        Args:
            path: Path to save model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """
        Load model from path
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])