from collections import deque
import random

class ReplayBuffer:
    """Memory buffer for storing and sampling experiences"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, log_prob, value):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return zip(*experiences)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()