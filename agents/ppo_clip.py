from agents.ppo_base import PPO

class PPO_CLIP(PPO):
    """
    PPO implementation with clipped objective constraint
    """
    def __init__(self, *args, **kwargs):
        # Override constraint_type to ensure it's 'clip'
        kwargs['constraint_type'] = 'clip'
        super(PPO_CLIP, self).__init__(*args, **kwargs)
    
    # Note: No need to override update_policy as we're using the
    # base implementation which already has the clipped objective