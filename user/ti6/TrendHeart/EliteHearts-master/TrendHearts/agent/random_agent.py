from agent import BaseAgent
import random


class RandomAgent(BaseAgent):
    def __init__(self, player_name):
        super().__init__(player_name)

    def pick_card(self, data):
        candidate_cards = data['self']['candidateCards']
        return random.choice(candidate_cards)