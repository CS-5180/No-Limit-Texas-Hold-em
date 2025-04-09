from collections import deque
import random
import numpy as np

# class ReplayBuffer:
#     """Memory buffer for storing and sampling experiences"""
#     def __init__(self, capacity=10000):
#         self.buffer = deque(maxlen=capacity)
    
#     def add(self, state, action, reward, next_state, done, log_prob, value):
#         """Add experience to buffer"""
#         self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
#     def sample(self, batch_size):
#         """Sample a batch of experiences"""
#         experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
#         return zip(*experiences)
    
#     def __len__(self):
#         return len(self.buffer)
    
#     def clear(self):
#         """Clear the buffer"""
#         self.buffer.clear()

class ReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.buffer = []  # Start with an empty list
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done, log_prob, value):
        max_prio = self.max_priority if self.size > 0 else 1.0
        
        experience = (state, action, reward, next_state, done, log_prob, value)
        
        if self.size < self.capacity:
            # If buffer is not yet full, append the new experience
            self.buffer.append(experience)
            self.size += 1
        else:
            # If buffer is full, replace an existing experience
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)  # Ensure we don't try to sample more than available
        
        if self.size < batch_size:
            indices = list(range(self.size))
        else:
            # Sample based on priorities
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        
        # Unpack the samples into separate lists
        states, actions, rewards, next_states, dones, log_probs, values = zip(*samples)
        
        return (states, actions, rewards, next_states, dones, log_probs, values), indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.size = 0
        self.position = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0