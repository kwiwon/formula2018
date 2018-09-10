class Card:
    rank_to_int = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_to_int = {
        'S': 0, 'C': 1, 'H': 2, 'D': 3
    }

    rank_suit_to_int = {
        '2S': 0, '2C': 1, '2H': 2, '2D': 3,
        '3S': 4, '3C': 5, '3H': 6, '3D': 7,
        '4S': 8, '4C': 9, '4H': 10, '4D': 11,
        '5S': 12, '5C': 13, '5H': 14, '5D': 15,
        '6S': 16, '6C': 17, '6H': 18, '6D': 19,
        '7S': 20, '7C': 21, '7H': 22, '7D': 23,
        '8S': 24, '8C': 25, '8H': 26, '8D': 27,
        '9S': 28, '9C': 29, '9H': 30, '9D': 31,
        'TS': 32, 'TC': 33, 'TH': 34, 'TD': 35,
        'JS': 36, 'JC': 37, 'JH': 38, 'JD': 39,
        'QS': 40, 'QC': 41, 'QH': 42, 'QD': 43,
        'KS': 44, 'KC': 45, 'KH': 46, 'KD': 47,
        'AS': 48, 'AC': 49, 'AH': 50, 'AD': 51
    }

    def __init__(self, card_string):
        self._rank = card_string[0].upper()
        self._suit = card_string[1].upper()

    @property
    def rank_int(self):
        return Card.rank_to_int[self._rank]

    @property
    def rank_str(self):
        return self._rank

    @property
    def suit_int(self):
        return Card.suit_to_int[self._suit]

    @property
    def suit_str(self):
        return self._suit

    @property
    def unique_int(self):
        return Card.rank_suit_to_int[self.rank_str + self.suit_str]

    def __str__(self):
        return self.rank_str + self.suit_str

    def __repr__(self):
        return self.rank_str + self.suit_str

    def __eq__(self, other):
        if self is None:
            return other is None
        elif other is None:
            return False
        return self.rank_int == other.rank_int and self.suit_int == other.suit_int

    def __hash__(self):
        return hash(self.rank_int.__hash__() + self.suit_int.__hash__())