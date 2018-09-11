import sys
import time
import random
import copy
import math
from player_info import PlayerState
from state_info import TreeState

class MCTSPlayer(object):

    def __init__(self, input_name):
        self.num_to_index = {"2":1, "3":2, "4":4, "5":8, "6":16, "7":32, "8":64, "9":128,
                             "T":256, "J":512, "Q":1024, "K":2048, "A":4096}
        self.index_to_num = {1:"2", 2:"3", 4:"4", 8:"5", 16:"6", 32:"7", 64:"8", 128:"9",
                             256:"T", 512:"J", 1024:"Q", 2048:"K", 4096:"A"}
        self.suit_to_index = {"C":0, "D":1, "S":2, "H":3}
        self.index_to_suit = {0:"C", 1:"D", 2:"S", 3:"H"}
        ## system info
        # list of all players name, set at new_peer
        self.all_player_name = []
        # current round index, initial at new_deal, increase at new_round
        self.round_index = 0
        # limit MCTS simulation time, set before everytime doing MCTS
        self.start_time = 0.0

        ## stored info
        self.my_name = input_name
        # PlayerSate class of all players, initial at new_game
        self.player_info = {}
        # my hand card
        self.hand_card = []
        # current open card, for checking if player has no suit as first card at turn_end
        self.open_card = []
        # if expose 2H, update at expose_cards_end
        self.expose_heart = False
        # count belonger number of all cards, initial at first new_round, update at turn_end
        self.card_possible_count = {}
        
        ## tree state info
        self.root_state = None
        self.current_state = None

    def _set_start_time(self):
        self.start_time = time.time()

    def _card_to_index(self, card_string):
        return self.num_to_index[card_string[0]], self.suit_to_index[card_string[1]]

    def _index_to_card(self, n, s):
        return self.index_to_num[n]+self.index_to_suit[s]

    def runMCTS(self, time_limit):
        if self.round_index > 12: return   # no need to run with last one card
        self._set_start_time()
        mcts_time = 0
        while time.time()-self.start_time < time_limit:
            #s = time.time()
            mcts_time += 1
            # step 1: traverse game tree along most promising path
            # step 2: expand new tree branch at most promising node
            back_path = self.traverse_tree()
            # step 3: playout to end at new branch node
            all_player_score = self.playout_and_score(back_path[-1])
            # step 4: back propagation along most promising path
            self.back_propagation(back_path, all_player_score)
            #print ("This run costs {:.6f} sec".format(time.time()-s))
        print ("This time MCTS runs {0} simulation from {1} round_index.".format(mcts_time, self.round_index))

    def traverse_tree(self):
        gen_fail_time = 0
        while not self._gen_all_player_hand_card():
            gen_fail_time += 1
        #print ("Gen card failed {0} times!!!".format(gen_fail_time))
        root = self.current_state
        tree_path = [root]
        while True:
            player_name = root.player_order[len(root.open_card)]
            candi_list = self._gen_candidate_list(root, self.player_info[player_name])
            (pick_c, to_s) = self._pick_biggest_UCT(root, candi_list, 1.41421356)
            self.player_info[player_name].del_one_hand_card(pick_c)
            stop = False
            if to_s is None: # expand new tree branch
                new_s = root.gen_next_state(pick_c, True)
                root = new_s
                stop = True
            else: # stop traverse if game over
                root = to_s
                stop = True if root.round_index==14 else False
            tree_path.append(root)  
            if stop: break
        return tree_path
        
    def _gen_all_player_hand_card(self):
        open_len = len(self.open_card)
        count = copy.copy(self.card_possible_count)
        # gen cards for other players
        for i in random.sample([0,1,2,3],4):
            if self.current_state.player_order[i] == self.my_name:
                self.player_info[self.my_name].set_gen_hand_card(copy.copy(self.hand_card))
                continue
            state = self.player_info[self.current_state.player_order[i]]
            remain_num = 13-self.round_index+(1 if i>=open_len else 0)
            must_have, may_have = [], []
            for j in (0,1,2,3):
                bit_mask = 1
                while bit_mask <= 4096:
                    c = self._index_to_card(bit_mask, j)
                    if state.may_have_card[j] & bit_mask:
                        if count[c] == 1:
                            must_have.append(c)
                        elif count[c] > 1:
                            may_have.append(c)
                        count[c] -= 1
                    bit_mask <<= 1
            try: # gen may fail under this logic
                gen_have = must_have + random.sample(may_have, remain_num-len(must_have))
            except:
                return False
            for j in gen_have: count[j] = 0
            state.set_gen_hand_card(gen_have)
        return True 

    def _gen_candidate_list(self, state, player):
        cand = []
        # if player has 2C
        if player.gen_hand_card[0]&1:
            cand = ["2C"]
        # if not leading player and have cards with suit as first card
        elif len(state.open_card) and player.gen_hand_card[self.suit_to_index[state.open_card[0][1]]]:
            for c in player.gen_hand:
                if c[1] == state.open_card[0][1]: cand.append(c)
        # if is leading player or have no card with suit as first card
        else:
            for c in player.gen_hand:
                # if this card is heart and ((is bleeding) or (not first round))
                # special case: first round with all hand cards are penalty cards
                if c[1]=="H" and (state.past_card[3] or (state.round_index!=1 or 
                    (player.gen_hand_card[0]==0 and player.gen_hand_card[1]==0 and 
                    (player.gen_hand_card[2]==1024 or player.gen_hand_card[3]==8191)))):
                    cand.append(c)
                # QS cannot be played in first round
                elif c=="QS" and state.round_index!=1:
                    cand.append(c)
                else:
                    cand.append(c)
        return cand

    def _pick_biggest_UCT(self, state, candi_list, C):
        big_value = -9999.999999
        big_list = []
        for i in candi_list:
            s = state.child.get(i, None)
            # UCT = average score + C*sqrt( ln(visit count of now state)/(visit count of that branch) )
            v = C*math.sqrt(math.log(state.visit_count)) if (s is None) else \
                    (float(s.all_sum_score[self.my_name])/s.visit_count + 
                    C*math.sqrt(math.log(state.visit_count)/s.visit_count))
            if v > big_value+0.000001: # choose biggest UCT
                big_value = v
                big_list = [(i,s)]
            elif -0.000001 < v-big_value < 0.000001: # if draw
                big_list.append((i,s))
        return random.sample(big_list, 1)[0]

    def playout_and_score(self, start_state):
        root = start_state
        # playout to end
        while root.round_index < 14:
            player_name = root.player_order[len(root.open_card)]
            candi_list = self._gen_candidate_list(root, self.player_info[player_name])
            pick = self.choose_card_policy(candi_list)
            self.player_info[player_name].del_one_hand_card(pick)
            root = root.gen_next_state(pick, False)
        # calculate all players score of this state
        ret = root.calculate_all_score(self.expose_heart)
        return ret

    # default is randomly choose from candidate list
    def choose_card_policy(self, candi_list):
        return random.sample(candi_list, 1)[0]

    def _calculate_score(self, get_card):
        score, b = 0, 1
        heart_score = -2 if self.expose_heart else -1
        while b <= 4096:
            if get_card[3] & b:
                score += heart_score
            b <<= 1
        if get_card[2] & 1024: #QS
            score += (-13)
        if get_card[0] & 256: #TC
            score *= 2
        if get_card[3] == 8191 and (get_card[2]&1024): # shooting the moon
            score *= (-4)
        return score

    def back_propagation(self, back_path, all_player_score):
        for s in back_path:
            s.visit_count += 1
            for (k, v) in all_player_score.items():
                s.all_sum_score[k] += v

    # call once at first new_round
    def _calculate_card_possible_count(self):
        # initialize
        self.card_possible_count = {}
        for i in "CDSH":
            for j in "23456789TJQKA":
                self.card_possible_count[j+i] = 0
        # counting, except for myself
        for (name,state) in self.player_info.items():
            if name == self.my_name: continue
            for i in (0,1,2,3):
                bit_mask = 1
                while bit_mask <= 4096:
                    c = self._index_to_card(bit_mask, i)
                    if state.may_have_card[i] & bit_mask:
                        self.card_possible_count[c] += 1
                    bit_mask <<= 1

    # aboves are methods for MCTS
    ##############################################################
    # follwings are methods for game protocal

    def set_all_player_name(self, data):
        self.all_player_name = []
        for i in data["players"]:
            self.all_player_name.append(i["playerName"])

    def process_new_deal(self, data):
        # reset round index
        self.round_index = 0
        # initial all_player_state
        self.player_info = {}
        for i in self.all_player_name:
            self.player_info[i] = PlayerState(i)
        # get my hand_card
        self.hand_card = data["self"]["cards"]
        # reset expose_heart
        self.expose_heart = False
        # delete my hand_card from may_have_card of other players
        for i in self.all_player_name:
            if i == self.my_name: continue
            for j in self.hand_card:
                self.player_info[i].del_possible_card(j)
        # initial current tree state, maybe you can load recorded game tree
        self.current_state = TreeState()
        self.current_state.visit_count = 1

    def choose_passing_cards(self, data):
        # default is randomly choose three cards
        pick = random.sample(self.hand_card, 3)
        receiver = data["receiver"]
        # set may_have_card of receiver
        for i in pick:
            self.player_info[receiver].set_possible_card(i)
        # delete cards from hand_card
        for i in pick:
            self.hand_card.remove(i)
        return pick

    def receive_opponent_cards(self, data):
        rec_card = data["self"]["receivedCards"]
        # delete may_have_card of other players
        for i in self.all_player_name:
            if i == self.my_name: continue
            for j in rec_card:
                self.player_info[i].del_possible_card(j)
        # add cards to hand_card
        self.hand_card.extend(rec_card)

    def expose_my_cards(self, data):
        candi_list = data["self"]["candidateCards"]
        # default is randomly expose
        if random.sample([True,False], 1)[0]:
            return candi_list
        else:
            return []

    def process_expose_cards(self, data):
        # delete may_have_card of players
        for i in data["players"]:
            if len(i["exposedCards"]) == 0: continue
            if "AH" in i["exposedCards"]: self.expose_heart = True
            player_name = i["playerName"]
            for j in self.all_player_name:
                if j == player_name: continue
                for k in i["exposedCards"]:
                    self.player_info[j].del_possible_card(k)

    def process_new_round(self, data):
        self.round_index += 1
        self.open_card = []
        order = data["roundPlayers"]
        # first round is important, we can know who has 2C and the leading player to show card
        if self.round_index == 1:
            for i in order[1:]:
                self.player_info[i].del_possible_card("2C")
            # initial card_possible_count
            self._calculate_card_possible_count()
            # initial player_order and other info
            self.current_state.player_order = order
            self.current_state.initial_all_sum_score()
            self.current_state.initial_all_get_card()
        # later round is for checking order is correctly updated in TreeState class
        else:
            if self.current_state.player_order != order:
                print ("{0}=/={1} order is wrongly updated!!!".format(
                          self.current_state.player_order, order))
            self.current_state.player_order = order
        # take any possible time to do MCTS
        self.runMCTS(0.2)

    # update info at turn_end
    def pick_card(self, data):
        candi_list = data["self"]["candidateCards"]
        self.runMCTS(0.7)
        big_value = -9999.999999
        big_list = []
        for i in candi_list:
            s = self.current_state.child.get(i, None)
            if s is None: continue # not choose branch which is never simulated
            v = float(s.all_sum_score[self.my_name])/s.visit_count
            if v > big_value+0.000001:
                big_value = v
                big_list = [i]
            elif -0.000001 < v-big_value < 0.000001:
                big_list.append(i)
        # if all candidate branches are never simulated, randomly choose from candidate
        return random.sample(big_list if len(big_list) else candi_list, 1)[0]

    def process_turn_end(self, data):
        turn_player = data["turnPlayer"]
        turn_card = data["turnCard"]
        if turn_player == self.my_name:
            self.hand_card.remove(turn_card)
        else:
            # clear card_possible_count
            self.card_possible_count[turn_card] = 0
            # delete may_have_card of players
            for i in self.all_player_name:
                self.player_info[i].del_possible_card(turn_card)
            # check if player has no suit as first card
            if len(self.open_card) and turn_card[1]!=self.open_card[0][1]:
                ret = self.player_info[turn_player].set_suit_to_zero(self.open_card[0][1])
                for i in ret:
                    self.card_possible_count[i] -= 1
        # append open_card
        self.open_card.append(turn_card)
        # state transition
        to_s = self.current_state.child.get(turn_card, None)
        if to_s is None:
            print ("Transit to never-simulated state {0}.".format(turn_card))
            to_s = self.current_state.gen_next_state(turn_card, True)
        self.current_state = to_s
        self.current_state.visit_count = 1
        # take any possible time to do MCTS
        self.runMCTS(0.2)

    def process_round_end(self, data):
        who_get = data["roundPlayer"]
        for i in self.open_card:
            self.player_info[who_get].set_get_card(i)
        # take any possible time to do MCTS
        self.runMCTS(0.7)

    def game_over(self, data):
        print ("Game Number:{0}".format(data["gameNumber"]))
        for i in data["players"]:
            print ("name:{0}\tgame_score:{1}\t".format(i["playerName"], i["gameScore"]) + 
                "deals:[{0},{1},{2},{3}]".format(i["deals"][0]["score"], i["deals"][1]["score"], 
                i["deals"][2]["score"], i["deals"][3]["score"]))




