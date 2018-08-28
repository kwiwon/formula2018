# coding=UTF-8
from PokerBot import PokerBot
from util.Card import Card

import json

class LowPlayBot(PokerBot):

    def __init__(self, name):
        super(LowPlayBot, self).__init__(name)
        self.my_hand_cards = []
        self.expose_card = False
        self.my_pass_card = []

    def receive_cards(self, data):
        self.my_hand_cards = self.get_cards(data)

    def pass_cards(self, data):
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)
        pass_cards = []
        count = 0

        for i in range(len(self.my_hand_cards)):
            card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card == Card("QS"):
                pass_cards.append(card)
                count += 1
            elif card == Card("TC"):
                pass_cards.append(card)
                count += 1

        for i in range(len(self.my_hand_cards)):
            card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card.suit_index == 2:
                pass_cards.append(card)
                count += 1
                if count == 3:
                    break

        if count < 3:
            for i in range(len(self.my_hand_cards)):
                card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
                if card not in self.game_score_cards:
                    pass_cards.append(card)
                    count += 1
                    if count == 3:
                        break

        return_values = []
        for card in pass_cards:
            return_values.append(card.toString())
        message = "Pass Cards:{}".format(return_values)
        self.my_pass_card = return_values
        return return_values

    def pick_card(self, data):

        received_cards = data['self']['receivedCards']
        picked_cards = data['self']['pickedCards']
        candidate_cards = data['self']['candidateCards']
        score_cards = data['self']['scoreCards']
        cards = data['self']['cards']

        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)

        message = "My Cards:{}".format(self.my_hand_cards)
        card_index = 0
        message = "Pick Card Event Content:{}".format(data)
        #print json.dumps(data)
        #print '------------------------------------------------------------'
        message = "Candidate Cards:{}".format(candidate_cards)
        print 'Candidate Cards:', json.dumps(candidate_cards)
        print 'Received Cards:', json.dumps(received_cards)
        print 'Picked Cards:', json.dumps(picked_cards)
        print 'Score Cards:', json.dumps(score_cards)
        print '------------------------------------------------------------'
        message = "Pick Card:{}".format(candidate_cards[card_index])
        print message
        print 'Pick Card Index', card_index, candidate_cards[card_index]
        print '============================================================'

        return candidate_cards[card_index]

    def expose_my_cards(self, yourcards):
        expose_card = []
        for card in self.my_hand_cards:
            if card == Card("AH"):
                expose_card.append(card.toString())
        message = "Expose Cards:{}".format(expose_card)
        return expose_card

    def expose_cards_end(self, data):
        players = data['players']
        expose_player = None
        expose_card = None
        for player in players:
            try:
                if player['exposedCards'] != [] and len(player['exposedCards']) > 0 \
                        and player['exposedCards'] is not None:
                    expose_player = player['playerName']
                    expose_card = player['exposedCards']
            except Exception, e:
                print e.message
        if expose_player is not None and expose_card is not None:
            message = "Player:{}, Expose card:{}".format(expose_player, expose_card)
            self.expose_card = True
        else:
            message = "No player expose card!"
            self.expose_card = False

    def receive_opponent_cards(self, data):
        self.my_hand_cards = self.get_cards(data)
        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                receive_cards = player['receivedCards']
                message = "User Name:{}, Picked Cards:{}, Receive Cards:{}".format(player_name, picked_cards,
                                                                                   receive_cards)

    def round_end(self, data):
        try:
            round_scores = self.get_round_scores(self.expose_card, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
        except Exception, e:
            print e.message

    def deal_end(self, data):
        self.my_hand_cards = []
        self.expose_card = False
        deal_scores, initial_cards, receive_cards, picked_cards = self.get_deal_scores(data)
        message = "Player name:{}, Pass Cards:{}".format(self.player_name, self.my_pass_card)

        for key in deal_scores.keys():
            message = "Player name:{}, Deal score:{}".format(key, deal_scores.get(key))

        for key in initial_cards.keys():
            message = \
                "Player name:{}, Initial cards:{}, Receive cards:{}, Picked cards:{}".format(key,
                                                                                             initial_cards.get(key),
                                                                                             receive_cards.get(key),
                                                                                             picked_cards.get(key))
    def game_over(self, data):
        game_scores = self.get_game_scores(data)
        for key in game_scores.keys():
            message = "Player name:{}, Game score:{}".format(key, game_scores.get(key))

    def pick_history(self, data, is_timeout, pick_his):
        for key in pick_his.keys():
            message = "Player name:{}, Pick card:{}, Is timeout:{}".format(key, pick_his.get(key), is_timeout)