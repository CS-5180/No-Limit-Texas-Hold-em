import numpy as np
import rlcard
from typing import Dict, Tuple, List, Any, Optional


class RLCardPokerEnv:
    """
    Wrapper for RLCard No-Limit Texas Hold'em environment.
    Standardizes the interface and handles state/action conversions.
    """

    def __init__(self, num_players=2, render_mode=None):
        """
        Initialize the poker environment.

        Args:
            num_players: Number of players in the game
            render_mode: Mode for rendering ('human' or None)
        """
        # Configure the environment
        self.num_players = num_players
        self.render_mode = render_mode

        # Create the environment
        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'game_num_players': num_players,
                'chips_for_each': 10000,  # Starting chips
                'dealer_id': 0,  # First player is dealer
            }
        )

        # Extract environment info
        self.num_actions = self.env.num_actions
        self.state_shape = self.env.state_shape

        # Define observation and action spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()

        # Game state tracking
        self.current_state = None
        self.current_player = None
        self.episode_rewards = [0] * num_players
        self.step_count = 0

    def _define_observation_space(self):
        """Define the standardized observation space structure."""
        # RLCard uses a dict observation space with specific structure
        return {
            'shape': self.state_shape,
            'dtype': np.float32
        }

    def _define_action_space(self):
        """Define the action space with discrete and continuous components."""
        # Action space for No-Limit Hold'em in RLCard:
        # Actions 0-5: fold, check, call, raise pot x1, raise pot x2, all-in
        return {
            'n': self.num_actions,
            'dtype': np.int32
        }

    def reset(self):
        """
        Reset the environment to start a new episode.

        Returns:
            Initial state observation
        """
        self.step_count = 0
        self.episode_rewards = [0] * self.num_players

        # Reset RLCard environment
        init_state, player_id = self.env.reset()

        # Store current state and player
        self.current_state = init_state
        self.current_player = player_id

        # Convert to standardized format
        observation = self._encode_state(init_state)

        return observation

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action ID (integer) to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1

        # Convert to RLCard action format if necessary
        rlcard_action = self._decode_action(action)

        # Take action in environment
        next_state, player_id = self.env.step(rlcard_action)

        # Check if the hand is over
        done = self.env.is_over()

        # Calculate rewards
        if done:
            # Get final payoffs at the end of the game
            payoffs = self.env.get_payoffs()
            reward = payoffs[self.current_player]
            # Store rewards for all players
            self.episode_rewards = payoffs
        else:
            # No intermediate rewards in poker
            reward = 0

        # Update current state and player
        self.current_state = next_state
        self.current_player = player_id

        # Convert state to standardized format
        observation = self._encode_state(next_state)

        # Prepare info dict
        info = {
            'player_id': player_id,
            'legal_actions': next_state['legal_actions'],
            'raw_legal_actions': next_state['raw_legal_actions'],
            'step_count': self.step_count
        }

        if done:
            info['episode_rewards'] = self.episode_rewards

        return observation, reward, done, info

    def _encode_state(self, state: Dict) -> Dict:
        """
        Convert RLCard state to a standardized observation format.

        Args:
            state: RLCard state dict

        Returns:
            Standardized observation dict
        """
        # Extract relevant information from state
        obs = {
            # Extract the raw observation vector used by RLCard
            'observation': state['obs'],

            # Additional useful information
            'player_id': self.current_player,
            'legal_actions': state['legal_actions'],
            'raw_obs': {
                'hand': state['raw_obs']['hand'],
                'public_cards': state['raw_obs']['public_cards'],
                'current_player': state['raw_obs']['current_player'],
                'pot': state['raw_obs']['pot'],
                'stakes': state['raw_obs']['stakes'],
            }
        }

        return obs

    def _decode_action(self, action: int) -> int:
        """
        Convert standardized action to RLCard action format.
        For basic actions, this is just the identity function since
        we're using RLCard's action IDs directly.

        Args:
            action: Action ID from agent

        Returns:
            Action ID for RLCard
        """
        # For now, we use RLCard's action space directly
        # This could be expanded to handle more complex action encoding
        return action

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the current state of the game using RLCard's utilities.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            Rendered frame if mode is 'rgb_array', else None
        """
        if self.current_state is None:
            return None

        raw_obs = self.current_state['raw_obs']

        # Use RLCard's built-in visualization utilities
        from rlcard.utils import print_card

        # Print divider and player turn
        print(f"\n============== No-Limit Texas Hold'em Poker ==============")
        print(f"=============== Player {self.current_player}'s turn ===============")

        # Print player's hand
        print("\nYour hand:")
        print_card(raw_obs['hand'])

        # Print public cards
        print("\nCommunity cards:")
        if raw_obs['public_cards']:
            print_card(raw_obs['public_cards'])
        else:
            print("None")

        # Print game state information
        print(f"\nPot: {raw_obs['pot']}")

        # Print player stakes
        if 'stakes' in raw_obs:
            print(f"Player 0 Chips: {raw_obs['stakes'][0]}")
            print(f"Player 1 Chips: {raw_obs['stakes'][1]}")

        # Print game phase based on number of community cards
        num_public_cards = len(raw_obs['public_cards'])
        if num_public_cards == 0:
            phase = "Pre-Flop"
        elif num_public_cards == 3:
            phase = "Flop"
        elif num_public_cards == 4:
            phase = "Turn"
        elif num_public_cards == 5:
            phase = "River"
        else:
            phase = "Unknown"
        print(f"Current phase: {phase}")

        # Print legal actions
        print("\nLegal actions:")
        for action_id in self.current_state['legal_actions']:
            action_str = self._action_to_string(action_id)
            print(f"{action_id}: {action_str}")

        # Add a divider at the end
        print("==========================================================\n")

        return None

    def _action_to_string(self, action_id: int) -> str:
        """Convert action ID to human-readable string."""
        # RLCard's No-Limit Hold'em actions
        if action_id == 0:
            return "Fold"
        elif action_id == 1:
            return "Check/Call"
        elif action_id == 2:
            return "Raise Pot"
        elif action_id == 3:
            return "Raise 2x Pot"
        elif action_id == 4:
            return "All-in"
        else:
            return f"Unknown action {action_id}"

    def close(self):
        """Clean up environment resources."""
        if hasattr(self.env, 'close'):
            self.env.close()