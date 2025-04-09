import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ppo_base import PPO

class PPO_KL(PPO):
    """
    PPO implementation with KL divergence constraint
    """
    def __init__(self, *args, **kwargs):
        # Override constraint_type to ensure it's 'kl'
        kwargs['constraint_type'] = 'kl'
        super(PPO_KL, self).__init__(*args, **kwargs)
        
        # Adaptive KL penalty coefficient
        self.kl_coef = 0.01
    
    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        """
        Update policy using KL constrained objective
        """
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old policy distribution
        with torch.no_grad():
            old_logits, _ = self.policy(states)
            old_probs = F.softmax(old_logits, dim=1)
        
        for _ in range(self.update_epochs):
            # Get current policy and value predictions
            logits, values = self.policy(states)
            values = values.squeeze()
            
            # Get current distribution
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute policy ratio and surrogate objective
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr_loss = -ratios * advantages
            
            # Compute KL divergence
            kl_div = torch.sum(old_probs * (torch.log(old_probs + 1e-8) - torch.log(probs + 1e-8)), dim=1).mean()
            
            # KL-constrained policy loss
            policy_loss = surr_loss.mean() + self.kl_coef * kl_div
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss with entropy bonus
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Store losses for metrics
            self.losses['policy_loss'].append(policy_loss.item())
            self.losses['value_loss'].append(value_loss.item())
            self.losses['entropy'].append(entropy.item())
            self.losses['total_loss'].append(loss.item())
        
        # Adaptively adjust KL coefficient based on observed KL
        # if kl_div < self.kl_target / 1.5:
        #     # KL too small, decrease penalty
        #     self.kl_coef /= 2
        # elif kl_div > self.kl_target * 1.5:
        #     # KL too large, increase penalty
        #     self.kl_coef *= 2
        kl_ratio = (kl_div / self.kl_target).cpu().item()
        self.kl_coef *= np.exp(0.1 * (kl_ratio - 1))
        self.kl_coef = np.clip(self.kl_coef, 1e-4, 100)