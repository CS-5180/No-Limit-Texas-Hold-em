"""
Configuration for PPO with clipped objective
"""
from configs.default_config import AGENT_CONFIG

# Extend the default config with CLIP-specific parameters
PPO_CLIP_CONFIG = {
    **AGENT_CONFIG,
    'clip_ratio': 0.2,  # Standard PPO clip ratio
}