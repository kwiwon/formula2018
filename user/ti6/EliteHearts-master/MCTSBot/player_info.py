import sys

class PlayerState(object):

    def __init__(self, input_name):
        self.num_to_index = {"2":1, "3":2, "4":4, "5":8, "6":16, "7":32, "8":64, "9":128,
                             "T":256, "J":512, "Q":1024, "K":2048, "A":4096}
        self.index_to_num = {1:"2", 2:"3", 4:"4", 8:"5", 16:"6", 32:"7", 64:"8", 128:"9",
                             256:"T", 512:"J", 1024:"Q", 2048:"K", 4096:"A"}
        self.suit_to_index = {"C":0, "D":1, "S":2, "H":3}
        ## stored info
        # player's name
        self.name = input_name
        # player's eaten card, bitmap of club/diamond/spade/heart
        self.get_card = [0,0,0,0]
        # possible cards of this player, bitmap of club/diamond/spade/heart
        self.may_have_card = [8191,8191,8191,8191]
        
        ## randomly generated cards of this player for running MCTS
        self.gen_hand_card = [0,0,0,0]
        self.gen_hand = []

    def _card_to_index(self, card_string):
        return self.num_to_index[card_string[0]], self.suit_to_index[card_string[1]]

    def set_possible_card(self, card_string):
        n, s = self._card_to_index(card_string)
        self.may_have_card[s] |= n

    def del_possible_card(self, card_string):
        n, s = self._card_to_index(card_string)
        if self.may_have_card[s] & n:
            self.may_have_card[s] ^= n

    def set_gen_hand_card(self, gen_list):
        self.gen_hand = gen_list
        self.gen_hand_card = [0,0,0,0]
        for i in gen_list:
            n, s = self._card_to_index(i)
            self.gen_hand_card[s] |= n

    def del_one_hand_card(self, pick_card):
        self.gen_hand.remove(pick_card)
        n, s = self._card_to_index(pick_card)
        self.gen_hand_card[s] ^= n

    def set_get_card(self, card_string):
        n, s = self._card_to_index(card_string)
        self.get_card[s] |= n

    # return original may_have_card of that suit
    def set_suit_to_zero(self, suit):
        ret, b, s = [], 1, self.suit_to_index[suit]
        while b <= 4096:
            if self.may_have_card[s] & b:
                self.may_have_card[s] ^= b
                ret.append(self.index_to_num[b]+suit)
            b <<= 1
        return ret

    def calculate_score(self, expose_heart):
        score, b = 0, 1
        heart_score = -2 if expose_heart else -1
        while b <= 4096:
            if self.get_card[3] & b:
                score += heart_score
            b <<= 1
        if self.get_card[2] & 1024: #QS
            score += (-13)
        if self.get_card[0] & 256: #TC
            score *= 2
        if self.get_card[3] == 8191 and (self.get_card[2]&1024): # shooting the moon
            score *= (-4)
        return score

    def print_debug(self):
        print ("name:{0} may_have_card:{1} get_card:{2} gen_hand:{3}".format(
            self.name, self.may_have_card, self.get_card, self.gen_hand))


