import rlcard

# Create the No Limit Texas Hold'em environment with a fixed seed for reproducibility.
env = rlcard.make('no-limit-holdem', config={
    'seed': 42,            # Ensures that results are reproducible
    'allow_step_back': False  # Disables the step back feature
})

# Start a new game, which returns the initial game state and the ID of the current player.
state, current_player = env.init_game()

# (Optional) Display initial state
print("Initial State:", state)
print("Current Player ID:", current_player)