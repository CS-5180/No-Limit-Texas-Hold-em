from env_wrapper.rlcard_setup import RLCardEnv
from logs.logger import Logger

"""
Class implementing the Advantage Actor Critic algorithm.
"""

class A2C:
    def __init__(self):
        self.env = RLCardEnv()
        self.log = Logger(log_name="A2C-Algorithm")
        self.log.info("Initializing A2C Algorithm.")