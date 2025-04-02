import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PokerNetwork(nn.Module):
    """
    Neural network for PPO policy and value function
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(PokerNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        shared_features = self.shared(x)
        action_logits = self.policy(shared_features)
        state_values = self.value(shared_features)
        return action_logits, state_values
    
    def get_action(self, state, legal_actions, device='cpu'):
        """
        Get an action from policy given state and legal actions
        """
        # Convert state to tensor and get logits
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_logits, state_value = self.forward(state_tensor)
        
        # Mask illegal actions with large negative values
        action_mask = torch.ones_like(action_logits) * float('-inf')
        for action in legal_actions:
            action_mask[0, action] = 0
        
        masked_logits = action_logits + action_mask
        
        # Sample from the probability distribution
        action_probs = F.softmax(masked_logits, dim=1)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Return action, log probability, and value
        return action.item(), dist.log_prob(action), state_value