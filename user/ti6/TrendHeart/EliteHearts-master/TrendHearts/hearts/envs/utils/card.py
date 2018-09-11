from treys import Card


class MyCard:
    """ Wrapper of treys's Card """

    def __init__(self, rank_suit):
        if isinstance(rank_suit, str):  # like Qs, Kc
            self._card = Card.new(rank_suit)
        elif isinstance(rank_suit, int):  # unique value from card
            self._card = rank_suit

    def __lt__(self, other):
        return self._card < other._card

    def __ge__(self, other):
        return self._card > other._card

    def __le__(self, other):
        return self._card <= other._card

    def __eq__(self, other):
        return self._card == other._card

    def __ne__(self, other):
        return self._card != other._card

    def __str__(self):
        return Card.int_to_pretty_str(self._card)

    def to_string(self):
        return self.rank_str.upper() + self.suit.upper()

    @property
    def rank_str(self):
        rank_int = Card.get_rank_int(self._card)
        return Card.STR_RANKS[rank_int]

    @property
    def rank_int(self):
        rank_int = Card.get_rank_int(self._card)
        return rank_int

    @property
    def suit(self):
        suit_int = Card.get_suit_int(self._card)
        return Card.INT_SUIT_TO_CHAR_SUIT[suit_int]

    @property
    def card(self):
        return self._card

    def __hash__(self):
        return hash(self.rank_str.__hash__()  + self.suit.__hash__())