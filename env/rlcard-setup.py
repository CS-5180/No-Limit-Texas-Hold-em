import rlcard
from logs.logger import Logger

# Create the No Limit Texas Hold'em environment with a fixed seed for reproducibility.

log = Logger(log_name="env")
log.info("Starting rlcard setup")

env = rlcard.make('no-limit-holdem', config={
    'seed': 42,            # Ensures that results are reproducible
    'allow_step_back': False  # Disables the step back feature
})

state = env.reset()
print("Initial state:", state)