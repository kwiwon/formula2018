
from .card import Card, SCORE_CARDS


class PokerBot(object):

    def __init__(self,player_name, system_log):
        self.round_cards_history=[] # list<Card> that all players sent out
        self.players = []
        self.pick_his={}            # map<player: string, cards: list<Card>> records pick card history of each player
        self.round_cards = {}       # map<player: string, card: string> records state of round
        self.round_suit = None
        self.score_cards={}
        self.cur_board = []
        self.picked_cards=[]
        self.picked_cards_this_round = []
        self.heart_broken = False
        self.is_card_exposed = False
        self.player_index = {}
        self.player_name = player_name
        self.self_index = -1
        self.players_current_picked_cards=[] # record the cards I have, don't know how to use it
        self.game_score_cards = SCORE_CARDS
        self.system_log=system_log

    # @abstractmethod
    def pass_cards(self,data):
        err_msg = self.__build_err_msg("pass_cards")
        raise NotImplementedError(err_msg)
    def pick_card(self,data):
        err_msg = self.__build_err_msg("pick_card")
        raise NotImplementedError(err_msg)
    def expose_my_cards(self,yourcards):
        err_msg = self.__build_err_msg("expose_my_cards")
        raise NotImplementedError(err_msg)
    def round_end(self,data):
        err_msg = self.__build_err_msg("round_end")
        raise NotImplementedError(err_msg)
    def deal_end(self,data):
        err_msg = self.__build_err_msg("deal_end")
        raise NotImplementedError(err_msg)
    def game_over(self,data):
        err_msg = self.__build_err_msg("game_over")
        raise NotImplementedError(err_msg)
    def pick_history(self,data,is_timeout,pick_his):
        err_msg = self.__build_err_msg("pick_history")
        raise NotImplementedError(err_msg)

    def reset_card_his(self):
        self.round_cards_history = []
        self.picked_cards = []
        self.pick_his={}

    def get_card_history(self):
        return self.round_cards_history

    def init_players(self, data, force_update=False):
        if force_update or self.players is None or len(self.players) == 0:
            players = data["players"]
            self.player_index = {}
            self.players = [{}, {}, {}, {}]
            for ind, (p, dp) in enumerate(zip(self.players, players)):
                p["playerName"] = dp["playerName"]
                p["playerNumber"] = dp["playerNumber"]
                p["hasCards"] = [Card(c) for c in dp["cards"]] if "cards" in dp else []
                p["lackCards"] = set()
                p["scoreCards"] = []
                p["score"] = 0
                if p["playerName"] == self.player_name:
                    p["self"] = True
                    self.self_index = ind
                else:
                    p["self"] = False
                self.player_index[p["playerName"]] = ind
        self.system_log.logger.debug("init players end: {}".format(self.players))

    def new_deal(self, data):
        self.reset_card_his()
        self.heart_broken = False
        self.is_card_exposed = False
        self.init_players(data, force_update=True)

    def new_round(self, data):
        self.cur_board = []
        self.picked_cards_this_round = []
        self.round_suit = None

    def update_pass_cards(self, deal, my_pass_card):  # call this method at child's pass_cards' end
        if deal in [1, 2, 3]:
            if deal == 1:
                ind = (self.self_index + 1) % 4
            elif deal == 2:
                ind = (self.self_index - 1 + 4) % 4
            else:  # deal == 3
                ind = (self.self_index + 2) % 4
            self.players[ind]["hasCards"].extend(my_pass_card)

    def receive_opponent_cards(self, data):
        self.init_players(data)
        cards = data["players"][self.self_index]["cards"]
        self.players[self.self_index]["hasCards"] = [Card(c) for c in cards]
        self.system_log.logger.debug(
            "receive_opponent_cards end: {}".format(self.players))

    def expose_cards_end(self, data):
        self.init_players(data)
        players = data["players"]
        for p in players:
            self.is_card_exposed |= len(p["exposedCards"]) > 0
            ind = self.player_index[p["playerName"]]
            cards = [Card(c) for c in p["exposedCards"]
                     if Card(c) not in self.players[ind]["hasCards"]]
            self.players[ind]["hasCards"].extend(cards)
        self.system_log.logger.debug(
            "expose_cards_end end: {}".format(self.players))

    def turn_end(self, data):
        roundNumber = data["roundNumber"]
        turnCard=data['turnCard']
        turnPlayer=data['turnPlayer']
        players=data['players']
        is_timeout=data['serverRandom']
        for player in players:
            player_name=player['playerName']
            if player_name==self.player_name:
                current_cards=player['cards']
                for card in current_cards:
                    self.players_current_picked_cards.append(Card(card))
        self.round_cards[turnPlayer]=Card(turnCard)
        opp_pick={}
        opp_pick[turnPlayer]=Card(turnCard)
        if (self.pick_his.get(turnPlayer))!=None:
            pick_card_list=self.pick_his.get(turnPlayer)
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer]=pick_card_list
        else:
            pick_card_list = []
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer] = pick_card_list
        self.round_cards_history.append(Card(turnCard))
        self.pick_history(data,is_timeout,opp_pick)

        self.picked_cards.append((roundNumber, turnPlayer, Card(turnCard)))
        self.picked_cards_this_round.append((turnPlayer, Card(turnCard)))
        self.cur_board.append(Card(turnCard))
        self.init_players(data)
        if turnPlayer == self.player_name:
            self.players[self.self_index]["hasCards"].remove(Card(turnCard))
        if self.round_suit is None:
            self.round_suit = Card(turnCard).suit
        else:
            if self.round_suit != Card(turnCard).suit:
                ind = self.player_index[turnPlayer]
                self.players[ind]["lackCards"].add(self.round_suit)
                self.heart_broken |= (Card(turnCard).suit == "H")

    def get_cards(self,data):
        try:
            receive_cards=[]
            players=data['players']
            for player in players:
                if player['playerName']==self.player_name:
                    cards=player['cards']
                    for card in cards:
                        receive_cards.append(Card(card))
                    break
            return receive_cards
        except Exception as e:
            self.system_log.show_message(e.message)
            return None

    def round_end_update(self, data):  # call this method at child's round_end
        score_cards = [c for c in self.cur_board if c in self.game_score_cards]
        winner = data["roundPlayer"]
        ind = self.player_index[winner]
        self.players[ind]["scoreCards"].extend(score_cards)
        self.players[ind]["score"] = data["players"][ind]["dealScore"]


    def get_round_scores(self,is_expose_card=False,data=None):
        if data!=None:
            players=data['roundPlayers']
            picked_user = players[0]
            round_card = self.round_cards.get(picked_user)
            score_cards=[]
            for i in range(len(players)):
                card=self.round_cards.get(players[i])
                if card in self.game_score_cards:
                    score_cards.append(card)
                if round_card.suit_index==card.suit_index:
                    if round_card.value<card.value:
                        picked_user = players[i]
                        round_card=card
            if (self.score_cards.get(picked_user)!=None): # accumulate the scores of picked_user in current round
                current_score_cards=self.score_cards.get(picked_user)
                score_cards+=current_score_cards
            self.score_cards[picked_user]=score_cards
            self.round_cards = {}

        receive_cards={}
        for key in self.pick_his.keys():
            picked_score_cards=self.score_cards.get(key)
            round_score = 0
            round_heart_score=0
            is_double = False
            if picked_score_cards!=None:
                for card in picked_score_cards:
                    if card in self.game_score_cards:
                        if card == Card("QS"):
                            round_score += -13
                        elif card == Card("TC"):
                            is_double = True
                        else:
                            round_heart_score += -1
                if is_expose_card:
                    round_heart_score*=2
                round_score+=round_heart_score
                if is_double:
                    round_score*=2
            receive_cards[key] = round_score
        return receive_cards

    def get_deal_scores(self, data):
        try:
            self.score_cards = {}
            final_scores  = {}
            initial_cards = {}
            receive_cards = {}
            picked_cards  = {}
            players = data['players']
            for player in players:
                player_name     = player['playerName']
                palyer_score    = player['dealScore']
                player_initial  = player['initialCards']
                player_receive  = player['receivedCards']
                player_picked   = player['pickedCards']

                final_scores[player_name] = palyer_score
                initial_cards[player_name] = player_initial
                receive_cards[player_name]=player_receive
                picked_cards[player_name]=player_picked
            return final_scores, initial_cards,receive_cards,picked_cards
        except Exception as e:
            self.system_log.show_message(e.message)
            return None

    def get_game_scores(self,data):
        try:
            receive_cards={}
            players=data['players']
            for player in players:
                player_name=player['playerName']
                palyer_score=player['gameScore']
                receive_cards[player_name]=palyer_score
            return receive_cards
        except Exception as e:
            self.system_log.show_message(e.message)
            return None
