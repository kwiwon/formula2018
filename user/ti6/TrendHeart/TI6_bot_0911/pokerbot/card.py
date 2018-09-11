import itertools
import random
from typing import List, Set, Dict, Tuple, Optional


class Card:
    # from trend's sample bot
    # Takes in strings of the format: "AS", "TC", "6D"
    suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                       "7": 7, "8": 8, "9": 9}
    suit_index_dict = {"S": 0, "C": 1, "H": 2, "D": 3}
    value_suit_dict = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                       7: "7", 8: "8", 9: "9"}
    index_suit_dict = {0: "S", 1: "C", 2: "H", 3: "D"}
    val_string = "AKQJT98765432"

    def __init__(self, card_string):
        value, self.suit = card_string[0], card_string[1]
        self.value = Card.suit_value_dict[value]
        self.suit_index = Card.suit_index_dict[self.suit]

        # to speed up, calculate hash here
        self._hash = hash(self.value.__hash__() + self.suit.__hash__())
        self._str = Card.val_string[14 - self.value] + self.suit

    @staticmethod
    def create_deck() -> Set:
        # create for convenient
        s = set()
        lst = range(0, 52)
        for card_id in lst:
            suit = int(card_id / 13)
            val = (card_id % 13) + 2
            s.add(CM[Card.value_suit_dict[val] + Card.index_suit_dict[suit]])
        return s

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        if self is None:
            return other is None
        elif other is None:
            return False
        return self.value == other.value and self.suit == other.suit

    def __hash__(self):
        return self._hash


class CardDeck:
    def __init__(self, info):
        # players:
        #  self: boolean, is this myself.
        #  playerName: str
        #  playerNumber: int
        #  hasCards: List[Card], known cards in hand (exposed AH, passed cards, self's cards)
        #  lackCards: List[Str], lack suits of card
        # pickedCards: List[Tuple(roundNumber, playerName, card)], the history of user picked/hand off cards

        # for debug log convenient, cast type "str" to Card
        # comment it if performance is demanded
        for p in info['players']:
            cards = p['hasCards']
            for i in range(len(cards)):
                if isinstance(cards[i], str):
                    cards[i] = Card(cards[i])

        for i in range(len(info['pickedCards'])):
            p = info['pickedCards'][i]
            if isinstance(p[2], str):
                info['pickedCards'][i] = tuple([p[0], p[1], Card(p[2])])

        self.players: List = info['players']
        self.picked_cards: List = info['pickedCards']

        # verify info
        pc = set()
        for p in self.picked_cards:
            pc.add(p[2])
        assert (len(pc) == len(self.picked_cards)), 'duplicated cards in picked_cards'

        # remains_card - picked_cars - player.hasCards
        remains_card: Set[Card] = Card.create_deck()
        remains_card = remains_card - set([c[2] for c in self.picked_cards])
        remains_card = remains_card - set(itertools.chain(*[p['hasCards'] for p in self.players]))

        # shuffle
        self.deck_list = list(remains_card)
        random.shuffle(self.deck_list)

    def gen_cards_single(self, p, deck: Set) -> Tuple[Optional[List], Optional[Set]]:
        picked_cards = [x[2] for x in self.picked_cards if x[1] == p['playerName']]
        has_cards = p['hasCards']
        lack_cards = p['lackCards']
        still_has_cards = set(has_cards) - set(picked_cards)
        to_gen_count = 13 - len(picked_cards) - len(still_has_cards)
        assert (0 <= to_gen_count or to_gen_count <= 13), 'gen_cards_single fail, to_gen_count is incorrect'  # test

        sample = list()
        if (not p['self']) and (to_gen_count > 0):
            if len(deck) == to_gen_count:
                # gen cards for the last player
                sample.extend(deck)
            else:
                removed_cards = set()
                for draw_card in deck:
                    if draw_card.suit not in lack_cards:
                        sample.append(draw_card)
                        removed_cards.add(draw_card)
                    if len(sample) == to_gen_count:
                        break
                deck = deck - removed_cards

        if len(sample) != to_gen_count:
            return None, None

        sample.extend(still_has_cards)
        return sample, deck

    def gen_cards(self) -> List:
        # gen cards for the player with the order of lack cards
        for i in range(len(self.players)):
            self.players[i]['_idx'] = i
        lst = sorted(self.players, key=lambda x: len(x['lackCards']), reverse=True)

        ok = False
        samples = [None] * len(self.players)

        # in some corner case, it's difficult to gen cards obey all players' criteria
        # shuffle & retry to gen
        for retry in range(10):
            samples = [None] * len(self.players)

            if retry > 0:
                # retry & shuffle
                random.shuffle(self.deck_list)
            deck = set(self.deck_list)  # cast, copy

            for i in range(len(lst)):
                idx = lst[i]['_idx']
                samples[idx], deck = self.gen_cards_single(lst[i], deck)
                if samples[idx] is None:
                    print(f'sample count in incorrect: {retry}')
                    break  # to retry

            if len([x for x in samples if x is not None]) == len(self.players):
                ok = True
                break

        assert ok, 'gen_cards fail'
        return samples


_value_list = Card.suit_value_dict.keys()
_suit_list = Card.suit_index_dict.keys()
_card_text = [x + y for x, y in itertools.product(_value_list, _suit_list)]
CM: Dict = {x: Card(x) for x in _card_text}  # card map for convenient

SCORE_CARDS = {CM["QS"], CM["TC"], CM["2H"], CM["3H"], CM["4H"],
               CM["5H"], CM["6H"], CM["7H"], CM["8H"], CM["9H"],
               CM["TH"], CM["JH"], CM["QH"], CM["KH"], CM["AH"]}


