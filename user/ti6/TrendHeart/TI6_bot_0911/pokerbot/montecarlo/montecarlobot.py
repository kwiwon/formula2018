from pokerbot.pokerbot import PokerBot
from pokerbot.card import Card, CardDeck
from .simulator import Simulator

import time


class MonteCarloBot(PokerBot):

    def __init__(self,name, system_log):
        super(MonteCarloBot,self).__init__(name, system_log)
        self.my_hand_cards=[]
        self.my_pass_card=[]
        self.maxTime = 0.5
        self.simulator = Simulator(system_log)

    def new_deal(self,data):
        super(MonteCarloBot, self).new_deal(data)
        self.my_hand_cards=self.get_cards(data)

    def pass_cards(self,data):
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
        self.my_pass_card=return_values
        self.update_pass_cards(data['dealNumber'], pass_cards)
        return return_values

    def pick_card(self,data):
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
        message = "Pick Card:{}".format(candidate_cards[card_index])
        self.system_log.show_message(message)
        self.system_log.save_logs(message)

        curPlayers = []
        for player_name in data['roundPlayers']:
            if self.player_name == player_name:
                break
            else:
                curPlayers.append(self.player_index[player_name])

        card_scores = [0] * len(candidate_cards)
        scores = [self.players[i]["score"] for i in range(4)]
        score_cards = [self.players[i]["scoreCards"] for i in range(4)]

        sampleN = 0
        endTime = time.time() + self.maxTime
        while time.time() < endTime:

            samplePlayers = self.gen_sample()

            for i in range(len(candidate_cards)):
                card_scores[i] += self.simulator.run(curPlayers, \
                                            self.cur_board, self.heart_broken, \
                                            samplePlayers, Card(candidate_cards[i]), \
                                            self.player_index[self.player_name], scores, \
                                            score_cards, self.is_card_exposed) # do simulator
            sampleN += 1

        message="[MonteCarloBot] #samples: {}".format(sampleN)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)

        for i in range(len(card_scores)):
            if card_scores[card_index] < card_scores[i]:
                card_index = i

        message="[MonteCarloBot] card scores: {}".format(card_scores)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)

        return candidate_cards[card_index]

    def expose_my_cards(self,data):
        expose_card=[]
        for card in self.my_hand_cards:
            if card==Card("AH"):
                expose_card.append(str(card))
        message = "Expose Cards:{}".format(expose_card)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        return expose_card

    def round_end(self,data):
        try:
            round_scores=self.get_round_scores(self.is_card_exposed, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
                self.system_log.show_message(message)
                self.system_log.save_logs(message)
        except Exception as e:
            self.system_log.show_message(str(e))

    def deal_end(self,data):
        self.my_hand_cards=[]
        self.is_card_exposed = False
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
        # Actually, local pick_his only contains one player who sent out card this round
        # self.pick_his (class level) records all players history
        for key in pick_his.keys():
            message = "Player name:{}, Pick card:{}, Is timeout:{}".format(key,pick_his.get(key),is_timeout)
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
        for i in range(len(self.players)):
            message = "[MonteCarloBot] {} hasCard: {}".format(i, self.players[i]["hasCards"])
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
            message = "[MonteCarloBot] {} lackCard: {}".format(i, self.players[i]["lackCards"])
            self.system_log.show_message(message)
            self.system_log.save_logs(message)

    def gen_sample(self):
        info = {"players": self.players, "pickedCards": self.picked_cards}
        self.system_log.logger.debug("info for cardDeck = {}".format(str(info)))
        deck = CardDeck(info)
        samples = deck.gen_cards()
        return samples

