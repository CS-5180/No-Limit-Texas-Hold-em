"""
Default configuration for PPO agents
"""

# Environment configuration
ENV_CONFIG = {
    'num_players': 2,
    'render_mode': None
}

# Agent base configuration
AGENT_CONFIG = {
    'hidden_dim': 128,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'update_epochs': 4,
    'minibatch_size': 64,
    'device': 'cpu'  # Will be overridden if CUDA is available
}

# Training configuration
TRAINING_CONFIG = {
    'episodes': 1000,
    'max_steps': 200,
    'update_frequency': 128,
    'eval_frequency': 50,
    'render_eval': False,
    'seed': 42
}

# Ablation study configuration
ABLATION_CONFIG = {
    'save_dir': 'models',
    'log_dir': 'logs',
    'figure_dir': 'figures'
}