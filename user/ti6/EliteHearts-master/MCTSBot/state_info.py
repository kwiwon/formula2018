import sys
import copy

class TreeState(object):

    def __init__(self):
        self.num_to_index = {"2":1, "3":2, "4":4, "5":8, "6":16, "7":32, "8":64, "9":128,
                             "T":256, "J":512, "Q":1024, "K":2048, "A":4096}
        self.suit_to_index = {"C":0, "D":1, "S":2, "H":3}
        ## stored info
        self.round_index = 1
        # appeared card in past, bitmap of club/diamond/spade/heart
        self.past_card = [0,0,0,0]
        # appearing card in this round, list of card string
        self.open_card = []
        # first time player_order is determined at new_round, later can be updated by itself
        self.player_order = []
        # record eaten cards of all players, name: [0,0,0,0]
        self.all_get_card = {}
        
        ## MCTS info
        self.visit_count = 0
        # back propagation of all players' score, name: score
        self.all_sum_score = {}
        # record child branch, card: to TreeState
        self.child = {}

    def _card_to_index(self, card_string):
        return self.num_to_index[card_string[0]], self.suit_to_index[card_string[1]]

    def _set_past_card(self, card_string):
        n, s = self._card_to_index(card_string)
        self.past_card[s] |= n

    def _append_open_card(self, card_string):
        self.open_card.append(card_string)

    def _win_open_index(self):
        if len(self.open_card)<4: return -1
        wn, ws = self._card_to_index(self.open_card[0])
        wd = 0
        for i in (1,2,3):
            n, s = self._card_to_index(self.open_card[i])
            if s==ws and n>wn:
                wn = n
                wd = i
        return wd

    def initial_all_get_card(self):
        for i in self.player_order:
            self.all_get_card[i] = [0,0,0,0]

    def initial_all_sum_score(self):
        for i in self.player_order:
            self.all_sum_score[i] = 0

    def _update_new_round(self, wd):
        self.round_index += 1
        # update all_get_card
        for i in (0,1,2,3):
            n, s = self._card_to_index(self.open_card[i])
            self.all_get_card[self.player_order[wd]][s] |= n
        # clear open_card
        self.open_card = []
        # reorder
        self.player_order = (self.player_order+self.player_order)[wd:wd+4]

    # generate new sate for expansion(do_link=True) and playout(do_link=False)
    def gen_next_state(self, card_string, do_link):
        s = TreeState()
        s.round_index = copy.copy(self.round_index)
        s.past_card = copy.copy(self.past_card)
        s._set_past_card(card_string)
        s.open_card = copy.copy(self.open_card)
        s._append_open_card(card_string)
        s.player_order = copy.copy(self.player_order)
        s.all_get_card = copy.deepcopy(self.all_get_card)
        s.initial_all_sum_score()
        win = s._win_open_index()
        if win != (-1): s._update_new_round(win)
        if do_link: self.child[card_string] = s # link child node
        return s

    def print_debug(self):
        print ("round:{0} past_card:{1} open_card:{2} visit_count:{3} all_sum_score:{4}".format(
            self.round_index, self.past_card, self.open_card, self.visit_count, self.all_sum_score))

    def calculate_all_score(self, expose_heart):
        ret_score = {}
        for (name,get_card) in self.all_get_card.items():
            score, b = 0, 1
            heart_score = -2 if expose_heart else -1
            while b <= 4096:
                if get_card[3] & b:
                    score += heart_score
                b <<= 1
            if get_card[2] & 1024:
                score += (-13)
            if get_card[0] & 256:
                score *= 2
            if get_card[3] == 8191 and (get_card[2]&1024): # shooting the moon
                score *= (-4)
            ret_score[name] = score
        return ret_score



