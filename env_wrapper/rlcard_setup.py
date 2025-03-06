import rlcard
from logs.logger import Logger

"""
Wrapper class for the RLCard environment.
"""
class RLCardEnv:
    def __init__(self):
        """
        Initialize the RLCard environment.
        Creates the No Limit Texas Hold'em environment with a fixed seed for reproducibility.
        """
        self.env = rlcard.make('no-limit-holdem', config={
            'seed': 42,            # Ensures that results are reproducible
            'allow_step_back': False  # Disables the step back feature
        })
        self.log = Logger(log_name="RLCardEnv")

        self.log.info("Starting up RLCardEnv")

    def reset(self):
        """
        Reset the environment and return an initial state.

        :return:
        """
        self.log.info("Resetting RLCardEnv")
        state = self.env.reset()
        return state