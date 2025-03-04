import rlcard

# Create the No Limit Texas Hold'em environment with a fixed seed for reproducibility.
env = rlcard.make('no-limit-holdem', config={
    'seed': 42,            # Ensures that results are reproducible
    'allow_step_back': False  # Disables the step back feature
})

state = env.reset()
print("Initial state:", state)