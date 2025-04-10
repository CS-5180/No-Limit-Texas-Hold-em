import torch
from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.ppo_clip import PPO_CLIP
from agents.ppo_kl import PPO_KL
import os


class HumanAgent:
    """
    Human agent that takes actions based on user input.
    """

    def __init__(self):
        self.name = "Human Player"

    def act(self, observation):
        """
        Ask human player for an action.

        Args:
            observation: Current game state observation

        Returns:
            int: Action selected by human player
        """
        legal_actions = observation['legal_actions']

        # Convert the OrderedDict keys to a list for easier selection
        action_codes = list(legal_actions.keys())

        # Display legal actions
        print("\nYour available actions:")
        for i, action_code in enumerate(action_codes):
            action_str = env._action_to_string(action_code)
            print(f"  {i}: {action_str} (action code: {action_code})")

        # Get human input
        while True:
            try:
                choice = int(input("\nEnter your choice (number): "))
                if 0 <= choice < len(action_codes):
                    # Return the action code, not the index
                    return action_codes[choice]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")


def play_poker_game(model_path, agent_type='ppo_clip', num_games=5):
    """
    Load a trained agent and play poker games against a human.

    Args:
        model_path: Path to saved model
        agent_type: Type of agent ('ppo_clip' or 'ppo_kl')
        num_games: Number of games to play
    """
    global env  # Make environment accessible to HumanAgent

    # Create environment
    env = RLCardPokerEnv(num_players=2, render_mode='human')

    # Get state and action dimensions
    observation = env.reset()
    state_dim = len(observation['observation'])
    action_dim = env.action_space['n']

    # Create trained agent
    if agent_type == 'ppo_clip':
        agent_class = PPO_CLIP
        agent_name = "PPO_CLIP Agent"
    else:
        agent_class = PPO_KL
        agent_name = "PPO_KL Agent"

    # Initialize trained agent
    trained_agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    # Use best model by default
    if model_path is None:
        # Look for best win rate model
        best_model_path = f"checkpoints/{agent_class.__name__}_best_winrate.pt"
        if os.path.exists(best_model_path):
            model_path = best_model_path
            print(f"Using best win rate model: {model_path}")
        else:
            model_path = f"models/{agent_class.__name__}.pt"
            print(f"Best model not found, using default path: {model_path}")

    # Load trained model
    print(f"Loading model from: {model_path}")
    trained_agent.load_model(model_path)

    # Create human player
    human_player = HumanAgent()

    print("\n=== Welcome to Poker vs AI ===")
    print(f"You'll be playing against a trained {agent_name}")
    print("The AI will be player 0, and you'll be player 1")

    # Play games
    for game in range(num_games):
        print(f"\n=== Starting Game {game + 1}/{num_games} ===")
        observation = env.reset()
        done = False
        ai_reward = 0
        human_reward = 0

        while not done:
            # Display current state
            print("\nCurrent state:")
            env.render()

            # Get current player
            current_player = observation['player_id']

            # Determine whose turn it is
            if current_player == 0:  # AI agent's turn
                # Get state and legal actions
                state = observation['observation']
                legal_actions = observation['legal_actions']

                # Get action from trained agent
                action, _, _ = trained_agent.select_action(state, legal_actions)

                # Print the action taken by the agent
                action_str = env._action_to_string(action)
                print(f"\n{agent_name} takes action: {action_str}")

                # Small pause to let user read AI's move
                input("Press Enter to continue...")

            else:  # Human player's turn
                # Get action from human player
                action = human_player.act(observation)

                # Print the action taken by the human
                action_str = env._action_to_string(action)
                print(f"\nYou chose: {action_str}")

            # Take action in environment
            next_observation, reward, done, info = env.step(action)

            # Update state and reward
            observation = next_observation
            if current_player == 0:
                ai_reward += reward
            else:
                human_reward += reward

        # Show game results
        print(f"\nGame {game + 1} finished!")
        print(f"Your total reward: {human_reward}")
        print(f"{agent_name} total reward: {ai_reward}")

        if human_reward > ai_reward:
            print("You won this game!")
        elif human_reward < ai_reward:
            print("The AI won this game!")
        else:
            print("The game ended in a tie!")

        if game < num_games - 1:
            play_again = input("\nReady for the next game? (y/n): ")
            if play_again.lower() != 'y':
                print("Thanks for playing!")
                break

    print("\nAll games completed!")


if __name__ == "__main__":
    # Example usage
    play_poker_game(
        model_path=None,
        agent_type="ppo_clip",
        num_games=3
    )