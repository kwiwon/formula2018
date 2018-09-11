import time
import math
import copy
import logging
import sys

from pokerbot.pokerbot import PokerBot
from pokerbot.card import Card, CardDeck, SCORE_CARDS

from .simulator import Simulator

logger = logging.getLogger("hearts_logs")


class HyperPara:
    c = 10
    n_iter = 20000
    max_run_time = 1  # in seconds


class Info:
    cur_board = []
    heart_broken = False
    is_card_exposed =False
    self_index = -1
    player_index = {}
    # {playerName: ind in players}, e.g.
    # {"player1": 1}
    players = []
    # list of dict, e.g.
    # [
    #   {
    #     "playerName": "player1"
    #     "hasCards": [Card(), Card]  #cards currently have
    #     "lackCards": set("H","D")
    #     "self": True
    #     "scoreCards": [Card(), Card()]   #the score card have
    #     "score": 0
    #     ""
    #    },
    # ]
    picked_cards = []
    # list of tuple(runoundNamer, playerName, Card)
    picked_cards_this_round = []
    # list of tuple(playerName, Card)

    # card_lack_count = 0  # copy from js
    # remaining_cards = [] # copy from js

    def __init__(self, bot):
        self.cur_board = copy.deepcopy(bot.cur_board)
        self.heart_broken = bot.heart_broken
        self.is_card_exposed = bot.is_card_exposed
        self.self_index = bot.self_index
        self.player_index = copy.deepcopy(bot.player_index)
        self.players = copy.deepcopy(bot.players)
        self.picked_cards = copy.deepcopy(bot.picked_cards)
        self.picked_cards_this_round = copy.deepcopy(bot.picked_cards_this_round)

    def __str__(self):
        return "Info(cur_board={}, heart_broken={}, is_card_exposed={}," \
               "self_index={},\n        players={},\n        picked_cards={}," \
               "    pick_cards_this_round={})".format(
            self.cur_board, self.heart_broken, self.is_card_exposed,
            self.self_index, self.players, self.picked_cards,
            self.picked_cards_this_round
        )

    @property
    def key(self):
        pc = "@".join(["{}|{}|{}".format(r, self.player_index[p], c)
                       for r, p, c in self.picked_cards])
        hb = "h" if self.heart_broken else "n"
        ce = "e" if self.is_card_exposed else "n"
        pl = "@".join(["{}#{}".format(
            "|".join(map(str, p["hasCards"])),
            "|".join(map(str, p["lackCards"]))) for p in self.players])
        return "{}-{}-{}-{}".format(pc, hb, ce, pl)

    def get_info_for_gen_sample(self):
        return {"players": self.players, "pickedCards": self.picked_cards}


class Node:
    count = 0
    value = 0
    info = None
    parent = None


class ActionNode(Node):
    card = None
    observations = {}
    is_root = False

    def __init__(self, parent_obs_node, action):
        if parent_obs_node is None:
            self.is_root = True
            self.observations = {}
        else:
            self.parent = parent_obs_node
            self.observations = {}
            assert isinstance(action, Card), "actions is not type of Card"
            self.card = action
            self.info = parent_obs_node.info  # ?

    def __str__(self):
        return "ActionNode(count={}, value={}, card={}, observation={}, is_root={})".format(
            self.count, self.value, self.card, ",".join(self.observations.keys()), self.is_root
        )

    def get_value(self):
        if self.count == 0:
            return 1
        else:
         return self.value + HyperPara.c * math.sqrt(
            math.log(self.parent.count) / float(self.count))

    def update_value(self, reward):
        self.value = (self.value * (self.value - 1) + reward) \
                     / (float(self.value) + sys.float_info.epsilon)

    def add_observation(self, observation_buffer, state=None, info=None):
        if not self.is_root:
            assert observation_buffer[0][1] == self.card, "obs[0][1] {} != self.card {}".format(observation_buffer[0][1], self.card)
        obs_hash = "|".join([str(self.info.player_index[o[0]]) + "-" + str(o[1])
                             for o in observation_buffer])
        if obs_hash not in self.observations:
            if info is not None:
                if self.is_root:
                    logger.info("init from root with {}".format(obs_hash))
                    # root node no need to init observation
                    self.observations[obs_hash] = ObservationNode(self, [])
                    # for the root node, no need to do rollout
                    # because this observation is not from root action
                    self.observations[obs_hash].init_actions()
                else:
                    logger.info("init obs with {} and info".format(obs_hash))
                    self.observations[obs_hash] = ObservationNode(self, observation_buffer)
                    self.observations[obs_hash].info = info
            if state is not None:
                logger.debug("init obs with {} and state".format(obs_hash))
                self.observations[obs_hash] = \
                    ObservationNode(self, observation_buffer, state)
        return obs_hash


class ObservationNode(Node):
    terminate = False
    actions = []
    obs = ""

    def __init__(self, parent_act_node, observation_buffer, state=None):
        if parent_act_node is not None:
            self.parent = parent_act_node
            self.obs = "|".join([str(parent_act_node.info.player_index[o[0]])
                                 + "-" + str(o[1]) for o in observation_buffer])
            self.actions = []
            info = copy.deepcopy(parent_act_node.info)
            if len(observation_buffer) > 0:
                assert self.parent.card == observation_buffer[0][1], "1st obs {} != parent action card {}".format(observation_buffer[0][1], self.parent.card)
            if state is not None:
                # update info from state: heart_broken, scoreCards, score
                info.heart_broken |= state.heart_broken
                for p, sp in zip(info.players, state.players):
                    p["scoreCards"] = sp["scoreCards"].copy()
                    p["score"] = sp["score"]

                n_cards_last_round = 4 - len(info.cur_board)

                # update self's hasCards,
                # the first observation should the card of current player
                info.players[info.self_index]["hasCards"].remove(observation_buffer[0][1])
                assert (set(info.players[info.self_index]["hasCards"]) == set(state.players[info.self_index]["hasCards"])), "hasCards has differnt cards: {} != {}".format(sorted(map(str,info.players[info.self_index]["hasCards"])), sorted(map(str,state.players[info.self_index]["hasCards"])))
                assert (len(info.players[info.self_index]["hasCards"]) == len(state.players[info.self_index]["hasCards"])), "hasCards length mismatch: {} != {}".format(sorted(map(str,info.players[info.self_index]["hasCards"])), sorted(map(str,state.players[info.self_index]["hasCards"])))

                # update lack cards
                last_round_cards = info.picked_cards_this_round \
                                   + observation_buffer[:n_cards_last_round]
                self._update_lack_card(last_round_cards, info)
                self._update_lack_card(observation_buffer[n_cards_last_round:], info)

                # update board, picked_cards info
                info.cur_board = [o[1] for o in observation_buffer[n_cards_last_round:]]
                info.picked_cards_this_round = observation_buffer[n_cards_last_round:]
                if len(info.picked_cards) > 0:
                    last_round_num = info.picked_cards[-1][0]
                else:
                    last_round_num = 0
                if n_cards_last_round == 4:
                    last_round_num += 1
                info.picked_cards.extend(
                    (last_round_num, o[0], o[1])
                    for o in observation_buffer[:n_cards_last_round])
                if len(observation_buffer) > n_cards_last_round:
                    info.picked_cards.extend(
                        (last_round_num + 1, o[0], o[1])
                        for o in observation_buffer[n_cards_last_round:])

                self.terminate = (last_round_num == 13)
            self.info = info

    def __str__(self):
        return "ObservationNode(count={}, value={}, obs={}, terminate={}, actions={}, info={})".format(
            self.count, self.value, self.obs, self.terminate, ",".join([str(a.card) for a in self.actions]), str(self.info)
        )

    def _update_lack_card(self, round_cards, info):
        if len(round_cards) > 0:
            round_suit = round_cards[0][1].suit
            for p, c in round_cards:
                if c.suit != round_suit:
                    info.players[info.player_index[p]]["lackCards"].add(
                        round_suit)

    def init_actions(self):
        me = self.info.players[self.info.self_index]
        if len(self.info.cur_board) > 0:
            suit = self.info.cur_board[0].suit
            suit_cards = [c for c in me["hasCards"] if c.suit == suit]
            if len(suit_cards) > 0:
                actions = suit_cards
            else:
                actions = me["hasCards"]
        elif len(me["hasCards"]) == 13:
            actions = [Card("2C")]
        elif self.info.heart_broken:
            actions = me["hasCards"]
        else:
            non_heart_cards = [c for c in me["hasCards"] if c.suit != "H"]
            if len(non_heart_cards) > 0:
                actions = non_heart_cards
            else:
                actions = me["hasCards"]
        self.actions = [ActionNode(self, a) for a in actions]


class State:  # complete information with sampled cards
    cur_board = []
    heart_broken = False
    is_card_exposed =False
    self_index = -1
    player_index = {}
    # {playerName: ind in players}, e.g.
    # {"player1": 1}
    players = []
    # list of dict, e.g.
    # [
    #   {
    #     "playerName": "player1"
    #     "hasCards": [Card(), Card]  #cards currently have
    #     "self": True
    #     "scoreCards": [Card(), Card()]   #the score card have
    #     "score": 0
    #     ""
    #    },
    # ]
    picked_cards_this_round = []
    # list of tuple(playerName, Card)

    def __init__(self, info, samples):
        self.cur_board = copy.deepcopy(info.cur_board)
        self.heart_broken = info.heart_broken
        self.is_card_exposed = info.is_card_exposed
        self.self_index = info.self_index
        self.player_index = copy.deepcopy(info.player_index)
        self.players = copy.deepcopy(info.players)
        self.picked_cards_this_round = copy.deepcopy(info.picked_cards_this_round)
        for p, s in zip(self.players, samples):
            p["hasCards"] = s

    def __str__(self):
        return "State(cur_board={}, heart_broken={},is_card_exposed={}," \
               "self_index={}, \n        players={}, \n" \
               "pick_cards_this_round={})".format(
            self.cur_board, self.heart_broken, self.is_card_exposed,
            self.self_index, self.players,
            self.picked_cards_this_round
        )


class PomDPBot(PokerBot):

    def __init__(self,name, system_log):
        super(PomDPBot,self).__init__(name, system_log)
        self.my_hand_cards = []
        self.expose_card = False
        self.my_pass_card = []
        self.root = {}
        self.cursor = None
        self.simulator = Simulator()
        self.observation_buffer = []

    def receive_cards(self, data):
        self.my_hand_cards = self.get_cards(data)

    def pass_cards(self, data):
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)
        pass_cards=[]
        count=0
        for i in range(len(self.my_hand_cards)):
            card=self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card==Card("QS"):
                pass_cards.append(card)
                count+=1
            elif card==Card("TC"):
                pass_cards.append(card)
                count += 1
        for i in range(len(self.my_hand_cards)):
            card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card.suit_index==2:
                pass_cards.append(card)
                count += 1
                if count ==3:
                    break
        if count <3:
            for i in range(len(self.my_hand_cards)):
                card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
                if card not in self.game_score_cards:
                    pass_cards.append(card)
                    count += 1
                    if count ==3:
                        break
        return_values=[]
        for card in pass_cards:
            return_values.append(str(card))
        message="Pass Cards:{}".format(return_values)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        self.my_pass_card=pass_cards
        self.update_pass_cards(data["dealNumber"], pass_cards)

        return return_values

    def new_deal(self, data):
        super(PomDPBot, self).new_deal(data)
        self.cursor = None
        self.observation_buffer = []

    def rollout(self, state, obs_node, depth):
        return self.simulator.run(state)

    def simulate(self, state, obs_node, depth):
        if obs_node.terminate:
            return 0
        if obs_node.actions is None or len(obs_node.actions) == 0:
            logger.debug(
                "rollout: depth={}, \n    obs_node={}, \n    state={}".format(
                    depth, str(obs_node), str(state)))
            obs_node.init_actions()
            logger.debug(
                "rollout: depth={}, \n    obs_node={}".format(
                    depth, str(obs_node), str(state)))
            return self.rollout(state, obs_node, depth)

        best_action = obs_node.actions[0]
        best_score = best_action.get_value()
        for a in obs_node.actions:
            score = a.get_value()
            if score > best_score:
                best_score = score
                best_action = a

        logger.debug(
            "depth={}, \n    obs_node={}, \n    state={}, \n    best_action={}".format(
                depth, str(obs_node), str(state), str(best_action)))

        simul_state, simul_obs, simul_score = self.simulator.step(state, best_action.card)
        assert best_action.card == simul_obs[0][1], "observation {} != best_action.card{}".format(best_action.card, simul_obs[0][1])

        logger.debug(
            "depth={}, \n    simul_state={}, \n    simul_obs={}, \n    obs_node={}, \n    state={}, \n    best_action={}".format(
                depth, str(simul_state), str(simul_obs), str(obs_node), str(state), str(best_action)))

        obs_hash = best_action.add_observation(simul_obs, state=simul_state)
        next_obs = best_action.observations[obs_hash]
        logger.debug(
            "next_obs={}".format(str(next_obs))
        )
        reward = simul_score \
                 + self.simulate(simul_state, next_obs, depth+1)

        obs_node.count += 1
        best_action.count += 1
        best_action.update_value(reward)
        return reward

    def gen_sample(self, obs_node):
        info = obs_node.info.get_info_for_gen_sample()
        logger.debug("info for cardDeck = {}".format(str(info)))
        for p in info["players"]:
            for c in p["hasCards"]:
                if not isinstance(c,Card):
                    logger.debug(p["playerName"], str(c), type(c))
        for (r,p,c) in info["pickedCards"]:
            if not isinstance(c, Card):
                logger.debug(str(r),p,str(c),type(c))
        deck = CardDeck(info)
        samples = deck.gen_cards()
        state = State(obs_node.info, samples)
        return state

    def search(self, candidate_cards):

        s = time.time()
        for i in range(HyperPara.n_iter):
            state = self.gen_sample(self.cursor)
            logger.debug(
                "i={}, \n    state={}, \n    cursor={}".format(i, str(state), str(self.cursor)))
            self.simulate(state, self.cursor, 0)
            e = time.time() - s
            if e > HyperPara.max_run_time:
                break
        message = "[PomDPBot] #samples: {}".format(i)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)

        actions = self.cursor.actions
        assert set([str(a.card) for a in actions]) == set(candidate_cards), "action, candidate cards mismatch: {}!= {}".format(sorted([str(a.card) for a in actions]), sorted(candidate_cards))
        assert len([str(a.card) for a in actions]) == len(candidate_cards), "action, candidate cards len mismatch: {}!= {}".format(sorted([str(a.card) for a in actions]), sorted(candidate_cards))
        actions = [a for a in actions if str(a.card) in candidate_cards]

        if actions:
            best_action = actions[0]
            best_score = best_action.value
            for a in actions:
                if a.value > best_score:
                    best_score = a.value
                    best_action = a
            self.cursor = best_action  # move root to next step
        else:
            raise Exception("no actions found")

        return best_action

    def pick_card(self, data):
        candidate_cards=data['self']['candidateCards']
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)
        message = "My Cards:{}".format(self.my_hand_cards)
        self.system_log.show_message(message)
        card_index=0
        message = "Pick Card Event Content:{}".format(data)
        self.system_log.show_message(message)
        message = "Candidate Cards:{}".format(candidate_cards)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)

        info = Info(self)
        if self.cursor is None:
            if info.key not in self.root:
                self.root[info.key] = ActionNode(None, None)
                self.root[info.key].info = info
            self.cursor = self.root[info.key]
            logger.debug("root contains: {}".format(",".join(self.root.keys())))
            logger.debug("cursor move to root: {}\n{}".format(info.key, str(self.cursor)))

        obs_hash = self.cursor.add_observation(
            self.observation_buffer, info=info)

        # take simulation result of from previous round's next step)
        self.cursor = self.cursor.observations[obs_hash]
        logger.debug(
            "cursor move to observation: {}\n{}".format(obs_hash, str(self.cursor)))
        self.observation_buffer = []

        action = self.search(candidate_cards)
        picked_card = str(action.card)

        message = "Pick Card:{}".format(picked_card)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        return picked_card

    def expose_my_cards(self, yourcards):
        expose_card=[]
        for card in self.my_hand_cards:
            if card==Card("AH"):
                expose_card.append(str(card))
        message = "Expose Cards:{}".format(expose_card)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        return expose_card

    def expose_cards_end(self,data):
        super(PomDPBot, self).expose_cards_end(data)
        players = data['players']
        expose_player=None
        expose_card=None
        for player in players:
            try:
                if player['exposedCards']!=[] and len(player['exposedCards'])>0 and player['exposedCards']!=None:
                    expose_player=player['playerName']
                    expose_card=player['exposedCards']
            except Exception as e:
                self.system_log.show_message(e)
                self.system_log.save_logs(e)
        if expose_player!=None and expose_card!=None:
            message="Player:{}, Expose card:{}".format(expose_player,expose_card)
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
            self.expose_card=True
        else:
            message="No player expose card!"
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
            self.expose_card=False

    def receive_opponent_cards(self,data):
        super(PomDPBot, self).receive_opponent_cards(data)
        self.my_hand_cards = self.get_cards(data)
        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                receive_cards = player['receivedCards']
                message = "User Name:{}, Picked Cards:{}, Receive Cards:{}".format(player_name, picked_cards,receive_cards)
                self.system_log.show_message(message)
                self.system_log.save_logs(message)

    def turn_end(self, data):
        super(PomDPBot, self).turn_end(data)
        turnCard=data['turnCard']
        turnPlayer=data['turnPlayer']
        self.observation_buffer.append((turnPlayer, Card(turnCard)))

    def round_end(self,data):
        try:
            self.round_end_update(data)
            round_scores=self.get_round_scores(self.expose_card, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
                self.system_log.show_message(message)
                self.system_log.save_logs(message)
        except Exception as e:
            self.system_log.show_message(e)

    def deal_end(self,data):
        self.my_hand_cards=[]
        self.expose_card = False
        deal_scores,initial_cards,receive_cards,picked_cards=self.get_deal_scores(data)
        message = "Player name:{}, Pass Cards:{}".format(self.player_name, self.my_pass_card)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        for key in deal_scores.keys():
            message = "Player name:{}, Deal score:{}".format(key,deal_scores.get(key))
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
        for key in initial_cards.keys():
            message = "Player name:{}, Initial cards:{}, Receive cards:{}, Picked cards:{}".format(key, initial_cards.get(key),receive_cards.get(key),picked_cards.get(key))
            self.system_log.show_message(message)
            self.system_log.save_logs(message)

    def game_over(self,data):
        game_scores = self.get_game_scores(data)
        for key in game_scores.keys():
            message = "Player name:{}, Game score:{}".format(key, game_scores.get(key))
            self.system_log.show_message(message)
            self.system_log.save_logs(message)

    def pick_history(self,data,is_timeout,pick_his):
        for key in pick_his.keys():
            message = "Player name:{}, Pick card:{}, Is timeout:{}".format(key,pick_his.get(key),is_timeout)
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
