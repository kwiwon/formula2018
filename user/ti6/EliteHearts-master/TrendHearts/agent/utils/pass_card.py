import random


def pass_random_card(data):
    player_hand_cards = data['self']['cards']
    return random.sample(player_hand_cards, 3)
