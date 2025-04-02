"""
Configuration for PPO with KL constraint
"""
from configs.default_config import AGENT_CONFIG

# Extend the default config with KL-specific parameters
PPO_KL_CONFIG = {
    **AGENT_CONFIG,
    'kl_target': 0.01,  # Target KL divergence
}