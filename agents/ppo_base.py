import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from models.neural_net import PokerNetwork
from utils.buffer import ReplayBuffer

class PPO(BaseAgent):
    """
    Base Proximal Policy Optimization implementation
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
        constraint_type='clip',
        kl_target=0.01,
        device='cpu'
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
            lr (float): Learning rate
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_ratio (float): PPO clip ratio
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy bonus coefficient
            max_grad_norm (float): Maximum gradient norm for clipping
            update_epochs (int): Number of update epochs per batch
            minibatch_size (int): Minibatch size for updates
            constraint_type (str): Type of constraint ('clip' or 'kl')
            kl_target (float): Target KL divergence for KL constraint
            device (str): Device to use ('cpu' or 'cuda')
        """
        super(PPO, self).__init__(state_dim, action_dim, device)
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.constraint_type = constraint_type
        self.kl_target = kl_target
        
        # Initialize policy network and optimizer
        self.policy = PokerNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = ReplayBuffer()
        
        # Initialize training metrics
        self.losses = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def select_action(self, state, legal_actions):
        """
        Select action from current policy
        
        Args:
            state: Current state
            legal_actions: List of legal actions
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state, legal_actions, self.device)
        
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in the buffer"""
        self.buffer.add(state, action, reward, next_state, done, log_prob, value)
    
    def compute_advantages(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Advantages and returns
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        # Compute GAE advantages going backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE formula
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # Compute returns (used for value function loss)
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        """
        Update policy using PPO objective
        
        This is the base implementation. Subclasses will override this
        method to implement specific constraint types.
        """
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            # Get current policy and value predictions
            logits, values = self.policy(states)
            values = values.squeeze()
            
            # Get log probs for actions taken
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratios and surrogate objectives
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # PPO clip objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss with entropy bonus
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Store losses for metrics
            self.losses['policy_loss'].append(policy_loss.item())
            self.losses['value_loss'].append(value_loss.item())
            self.losses['entropy'].append(entropy.item())
            self.losses['total_loss'].append(loss.item())
    
    def learn(self):
        """Learn from collected experiences"""
        # Check if we have enough data
        if len(self.buffer) < self.minibatch_size:
            return
        
        # Get all experiences
        # states, actions, rewards, next_states, dones, log_probs, values = zip(*self.buffer.buffer)
        (states, actions, rewards, next_states, dones, log_probs, values), indices, weights = self.buffer.sample(self.minibatch_size)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        log_probs = np.array(log_probs)
        values = np.array(values)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # Update policy
        self.update_policy(states, actions, log_probs, returns, advantages)
        
        # Clear buffer after learning
        self.buffer.clear()