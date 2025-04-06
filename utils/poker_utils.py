"""
This file was added because PPO variants were performing very poorly.
"""

def estimate_hand_strength(hand, community_cards):
    """
    Estimate the strength of a poker hand (0-1 scale)

    Args:
        hand: List of card strings (e.g., ['As', 'Kh'])
        community_cards: List of community card strings

    Returns:
        Float between 0-1 representing hand strength
    """
    if not hand:
        return 0.0

    # Extract ranks and suits
    card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    # Count cards by rank and suit
    all_cards = hand + community_cards
    ranks = [card[0] if card[0] != '1' else '10' for card in all_cards]
    suits = [card[-1] for card in all_cards]

    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    # Base strength on hand types
    strength = 0.0

    # High card value (higher cards are worth more)
    high_card_value = 0
    for card in hand:
        rank = card[0] if card[0] != '1' else '10'
        if rank in card_values and card_values[rank] > high_card_value:
            high_card_value = card_values[rank]

    strength += 0.05 * (high_card_value / 14)  # Normalize by max value (Ace)

    # Pair
    if max(rank_counts.values(), default=0) >= 2:
        strength += 0.25

    # Two pair or three of a kind
    if list(rank_counts.values()).count(2) >= 2 or max(rank_counts.values(), default=0) >= 3:
        strength += 0.2

    # Straight potential
    if len(set(ranks)) >= 5:
        strength += 0.1

    # Flush potential
    if max(suit_counts.values(), default=0) >= 4:
        strength += 0.2

    # Premium starting hands in holdem
    if len(community_cards) == 0:  # Preflop
        if 'A' in ranks:
            strength += 0.15
        if 'K' in ranks:
            strength += 0.1
        if 'A' in ranks and 'K' in ranks:
            strength += 0.1  # Bonus for AK
        if ranks[0] == ranks[1]:  # Pocket pair
            strength += 0.25

    return min(strength, 1.0)  # Cap at 1.0


def calculate_pot_odds(bet_to_call, pot_size):
    """
    Calculate pot odds (ratio of call amount to potential win)

    Args:
        bet_to_call: Amount needed to call
        pot_size: Current pot size

    Returns:
        Float representing pot odds (0-1)
    """
    if bet_to_call <= 0:
        return 0
    return bet_to_call / (pot_size + bet_to_call)


def shaped_reward(state, action, reward, next_state, done, info):
    """
    Apply reward shaping to improve learning signal

    Args:
        state: Current state dict
        action: Action taken
        reward: Raw reward from environment
        next_state: Next state dict
        done: Whether episode is done
        info: Additional info from environment

    Returns:
        Shaped reward
    """
    # Start with the original reward
    shaped_reward = reward

    # Only apply shaping for intermediate steps
    if not done:
        try:
            # Extract useful information
            raw_obs = state.get('raw_obs', {})
            hand = raw_obs.get('hand', [])
            community_cards = raw_obs.get('public_cards', [])
            pot_size = raw_obs.get('pot', 0)

            # Calculate bet to call (simplified)
            # In a real implementation, you'd need to track this properly
            bet_to_call = 1  # Simplified
            if 'legal_actions' in state and action in [1, 2, 3, 4]:
                bet_to_call = action * 2  # Very simplistic approximation

            # Calculate hand strength and pot odds
            hand_strength = estimate_hand_strength(hand, community_cards)
            pot_odds = calculate_pot_odds(bet_to_call, pot_size)

            # Reward/penalize based on action vs hand strength

            # Folding with weak hand is good
            if action == 0 and hand_strength < 0.3 and pot_odds > 0.2:
                shaped_reward += 0.05

            # Folding with strong hand is bad
            elif action == 0 and hand_strength > 0.6:
                shaped_reward -= 0.1

            # Calling with weak hand and bad pot odds is bad
            elif action == 1 and hand_strength < 0.3 and pot_odds > 0.3:
                shaped_reward -= 0.05

            # Raising with strong hand is good
            elif action in [2, 3] and hand_strength > 0.7:
                shaped_reward += 0.05

            # All-in with weak hand is bad
            elif action == 4 and hand_strength < 0.5:
                shaped_reward -= 0.15

            # All-in with very strong hand is good
            elif action == 4 and hand_strength > 0.8:
                shaped_reward += 0.1

        except Exception as e:
            # Fallback in case of error in reward shaping
            print(f"Error in reward shaping: {e}")
            shaped_reward = reward

    return shaped_reward