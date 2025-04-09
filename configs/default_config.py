"""
Default configuration for PPO agents
"""

import torch

# Environment configuration
ENV_CONFIG = {
    'num_players': 2,
    'render_mode': None
}

# # Agent base configuration
# AGENT_CONFIG = {
#     'hidden_dim': 512,
#     'lr': 3e-4,
#     'gamma': 0.99,
#     'gae_lambda': 0.95,
#     'value_coef': 0.5,
#     'entropy_coef': 0.02,
#     'max_grad_norm': 0.5,
#     'update_epochs': 8,
#     'minibatch_size': 246,
#     'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
# }

# # Training configuration
# TRAINING_CONFIG = {
#     'episodes': 1000,
#     'max_steps': 200,
#     'update_frequency': 256,
#     'eval_frequency': 500,
#     'render_eval': False,
#     'seed': 42,
#     'eval_episodes': 50
# }

# For AGENT_CONFIG in configs/default_config.py:
AGENT_CONFIG = {
    'hidden_dim': 256,
    'lr': 2.5e-4,
    'gamma': 0.995,
    'gae_lambda': 0.97,
    'value_coef': 0.5,
    'entropy_coef': 0.05,
    'max_grad_norm': 0.5,
    'update_epochs': 6,
    'minibatch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
}

# For TRAINING_CONFIG:
TRAINING_CONFIG = {
    'episodes': 10000,
    'max_steps': 200,
    'update_frequency': 64,
    'eval_frequency': 100,
    'render_eval': False,
    'seed': 42
}

# Ablation study configuration
ABLATION_CONFIG = {
    'save_dir': 'models',
    'log_dir': 'logs',
    'figure_dir': 'figures'
}