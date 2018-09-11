from treys import Deck
from hearts.envs.utils import Card


class MyDeck(Deck):
    def __init__(self):
        super().__init__()

    def draw(self, n=1):
        if n == 1:
            return Card(self.cards.pop(0))

        cards = []
        for i in range(n):
            cards.append(self.draw())
        return cards

    @property
    def size(self):
        return len(self.cards)

    def remove(self, card: Card):
        self.cards.remove(card.card)

