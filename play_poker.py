import torch
from env_wrapper.rlcard_setup import RLCardPokerEnv
from agents.ppo_clip import PPO_CLIP
from agents.ppo_kl import PPO_KL
from agents.random_agent import RandomAgent  # Import the random agent
import os


def play_poker_game(model_path, agent_type='ppo_clip', num_games=5):
    """
    Load a trained agent and play poker games.

    Args:
        model_path: Path to saved model
        agent_type: Type of agent ('ppo_clip' or 'ppo_kl')
        num_games: Number of games to play
    """
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

    # Create opponent (random agent)
    opponent = RandomAgent()

    # Play games
    for game in range(num_games):
        print(f"\n=== Starting Game {game + 1}/{num_games} ===")
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Display current state
            print("\nCurrent state:")
            env.render()

            # Get current player
            current_player = observation['player_id']

            # Determine whose turn it is
            if current_player == 0:  # Trained agent's turn
                # Get state and legal actions
                state = observation['observation']
                legal_actions = observation['legal_actions']

                # Get action from trained agent
                action, _, _ = trained_agent.select_action(state, legal_actions)

                # Print the action taken by the agent
                action_str = env._action_to_string(action)
                print(f"\n{agent_name} takes action: {action_str}")
            else:  # Opponent's turn
                # Get action from random opponent
                action = opponent.act(observation)

                # Print the action taken by the opponent
                action_str = env._action_to_string(action)
                print(f"\nOpponent takes action: {action_str}")

            # Take action in environment
            next_observation, reward, done, info = env.step(action)

            # Update state and reward
            observation = next_observation
            if current_player == 0:  # Only track rewards for trained agent
                total_reward += reward

            # Prompt user to continue
            input("\nPress Enter to continue to next action...")

        print(f"Game {game + 1} finished. {agent_name} total reward: {total_reward}")

    print("\nAll games completed!")


if __name__ == "__main__":
    # Example usage
    play_poker_game(
        model_path=None,
        agent_type="ppo_clip",
        num_games=3
    )