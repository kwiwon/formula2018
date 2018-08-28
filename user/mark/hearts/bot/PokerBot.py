# coding=UTF-8
from util.Card import Card

class PokerBot(object):

    def __init__(self, player_name):
        self.round_cards_history = []
        self.pick_his = {}
        self.round_cards = {}
        self.score_cards = {}
        self.player_name = player_name
        self.players_current_picked_cards = []
        self.game_score_cards = {Card("QS"), Card("TC"),
                                 Card("2H"), Card("3H"), Card("4H"), Card("5H"), Card("6H"), Card("7H"), Card("8H"),
                                 Card("9H"), Card("TH"), Card("JH"), Card("QH"), Card("KH"), Card("AH")}

    # @abstractmethod
    def receive_cards(self, data):
        err_msg = self.__build_err_msg("receive_cards")
        raise NotImplementedError(err_msg)

    def pass_cards(self, data):
        err_msg = self.__build_err_msg("pass_cards")
        raise NotImplementedError(err_msg)

    def pick_card(self, data):
        err_msg = self.__build_err_msg("pick_card")
        raise NotImplementedError(err_msg)

    def expose_my_cards(self, yourcards):
        err_msg = self.__build_err_msg("expose_my_cards")
        raise NotImplementedError(err_msg)

    def expose_cards_end(self, data):
        err_msg = self.__build_err_msg("expose_cards_announcement")
        raise NotImplementedError(err_msg)

    def receive_opponent_cards(self, data):
        err_msg = self.__build_err_msg("receive_opponent_cards")
        raise NotImplementedError(err_msg)

    def round_end(self, data):
        err_msg = self.__build_err_msg("round_end")
        raise NotImplementedError(err_msg)

    def deal_end(self, data):
        err_msg = self.__build_err_msg("deal_end")
        raise NotImplementedError(err_msg)

    def game_over(self, data):
        err_msg = self.__build_err_msg("game_over")
        raise NotImplementedError(err_msg)

    def pick_history(self, data, is_timeout, pick_his):
        err_msg = self.__build_err_msg("pick_history")
        raise NotImplementedError(err_msg)

    def reset_card_his(self):
        self.round_cards_history = []
        self.pick_his = {}

    def get_card_history(self):
        return self.round_cards_history

    def turn_end(self, data):
        turnCard = data['turnCard']
        turnPlayer = data['turnPlayer']
        players = data['players']
        is_timeout = data['serverRandom']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                current_cards = player['cards']
                for card in current_cards:
                    self.players_current_picked_cards.append(Card(card))
        self.round_cards[turnPlayer] = Card(turnCard)
        opp_pick = {}
        opp_pick[turnPlayer] = Card(turnCard)
        if (self.pick_his.get(turnPlayer)) != None:
            pick_card_list = self.pick_his.get(turnPlayer)
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer] = pick_card_list
        else:
            pick_card_list = []
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer] = pick_card_list
        self.round_cards_history.append(Card(turnCard))
        self.pick_history(data, is_timeout, opp_pick)

    def get_cards(self, data):
        try:
            receive_cards = []
            players = data['players']
            for player in players:
                if player['playerName'] == self.player_name:
                    cards = player['cards']
                    for card in cards:
                        receive_cards.append(Card(card))
                    break
            return receive_cards
        except Exception, e:
            print e.message
            return None

    def get_round_scores(self, is_expose_card=False, data=None):
        if data != None:
            players = data['roundPlayers']
            picked_user = players[0]
            round_card = self.round_cards.get(picked_user)
            score_cards = []
            for i in range(len(players)):
                card = self.round_cards.get(players[i])
                if card in self.game_score_cards:
                    score_cards.append(card)
                if round_card.suit_index == card.suit_index:
                    if round_card.value < card.value:
                        picked_user = players[i]
                        round_card = card
            if (self.score_cards.get(picked_user) != None):
                current_score_cards = self.score_cards.get(picked_user)
                score_cards += current_score_cards
            self.score_cards[picked_user] = score_cards
            self.round_cards = {}

        receive_cards = {}
        for key in self.pick_his.keys():
            picked_score_cards = self.score_cards.get(key)
            round_score = 0
            round_heart_score = 0
            is_double = False
            if picked_score_cards != None:
                for card in picked_score_cards:
                    if card in self.game_score_cards:
                        if card == Card("QS"):
                            round_score += -13
                        elif card == Card("TC"):
                            is_double = True
                        else:
                            round_heart_score += -1
                if is_expose_card:
                    round_heart_score *= 2
                round_score += round_heart_score
                if is_double:
                    round_score *= 2
            receive_cards[key] = round_score
        return receive_cards

    def get_deal_scores(self, data):
        try:
            self.score_cards = {}
            final_scores = {}
            initial_cards = {}
            receive_cards = {}
            picked_cards = {}
            players = data['players']
            for player in players:
                player_name = player['playerName']
                palyer_score = player['dealScore']
                player_initial = player['initialCards']
                player_receive = player['receivedCards']
                player_picked = player['pickedCards']

                final_scores[player_name] = palyer_score
                initial_cards[player_name] = player_initial
                receive_cards[player_name] = player_receive
                picked_cards[player_name] = player_picked
            return final_scores, initial_cards, receive_cards, picked_cards
        except Exception, e:
            print e.message
            return None

    def get_game_scores(self, data):
        try:
            receive_cards = {}
            players = data['players']
            for player in players:
                player_name = player['playerName']
                palyer_score = player['gameScore']
                receive_cards[player_name] = palyer_score
            return receive_cards
        except Exception, e:
            print e.message
            return None
