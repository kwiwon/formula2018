from ..pokerbot import Card

class Simulator:
    def __init__(self, system_log):
        self.curBoard = []                         # list<Card>
        self.heartBroken = False
        self.curPlayers = []                     # list<Integer>
        self.curCards = [[] for i in range(4)]     # list<list<Card>>
        self.curPlayer = 0
        self.nextFirst = -1
        self.scores = []
        self.score_cards = [[] for i in range(4)]
        self.is_expose_card = False
        self.game_score_cards = {Card("QS"), Card("TC"), Card("2H"), Card("3H"), Card("4H"), Card("5H"), Card("6H"),
                           Card("7H"), Card("8H"), Card("9H"), Card("TH"), Card("JH"), Card("QH"), Card("KH"),
                           Card("AH")}
        self.system_log = system_log

    # curPlayes : list<Integer>, is list of player's id
    # curBoard : List<Card>, is current cards on board
    # curCards : List<List<Card>>, is sample cards of each player
    # cardToPlay: Card, is card to play this round
    # scores: List<Integer>, is list of player's score
    def run(self, curPlayers, curBoard, heartBroken, curCards, cardToPlay, myID, scores, score_cards, is_expose_card):
        '''
        message = "[Simulator] curCards: {}".format(curCards)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        '''

        self.curPlayers.extend(curPlayers)
        self.curBoard.extend(curBoard)
        self.heartBroken = heartBroken
        for i in range(4):
            self.curCards[i].extend(curCards[i])
        self.curPlayer = myID
        self.scores = []
        self.scores.extend(scores)
        self.score_cards = [[] for i in range(4)]
        '''
        message = "[Simulator] before score cards: {}".format(score_cards)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        '''
        for i in range(4):
            self.score_cards[i].extend(score_cards[i])
        self.is_expose_card = is_expose_card

        self._play(self.curPlayer, cardToPlay)
        self._rollout()

        '''
        message = "[Simulator] after score_cards : {}".format(self.score_cards)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        message = "[Simulator] scores: {}".format(self.scores)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        '''

        moonShooter = -1
        for i in range(4):
            if len(self.score_cards[i]) == 15:
                moonShooter = i
            elif len(self.score_cards[i]) == 14 and Card("TC") not in self.score_cards[i]:
                moonShooter = i

        if moonShooter != -1:
            '''
            message = "[Simulator] moonShooter : {}".format(moonShooter)
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
            message = "[Simulator] ms score_cards : {}".format(self.score_cards[moonShooter])
            self.system_log.show_message(message)
            self.system_log.save_logs(message)
            '''
            if moonShooter == myID:
                return (self.scores[moonShooter] * -1) * 4
            else:
                return (self.scores[moonShooter]) * 4

        return self.scores[myID]

    def _play(self, player, card):
        self.curPlayers.append(player)
        self.curBoard.append(card)

        if card.suit_index == 2:
            self.heartBroken = True

        self.curCards[player].remove(card)

    def _get_candidate_cards(self, cards):
        candidate_cards = []
        if not self.curBoard:
            if self.heartBroken:
                candidate_cards.extend(cards)
            else:
                for card in cards:
                    if card.suit_index != 2:
                        candidate_cards.append(card)
        else:
            suit = self.curBoard[0].suit_index
            for card in cards:
                if card.suit_index == suit:
                    candidate_cards.append(card)
        if len(candidate_cards) == 0:
            candidate_cards.extend(cards)
        return candidate_cards

    def _rollout(self):
        self.curPlayer = (self.curPlayer +1) %4

        while len(self.curBoard) < 4:
            self._play(self.curPlayer, self._others_pick_card(self.curCards[self.curPlayer]))
            self.curPlayer = (self.curPlayer +1) %4
        self._end_round()

        while len(self.curCards[self.nextFirst]) > 0:
            self.curPlayer = self.nextFirst
            while len(self.curBoard) < 4:
                self._play(self.curPlayer, self._others_pick_card(self.curCards[self.curPlayer]))
                self.curPlayer = (self.curPlayer +1) %4
            self._end_round()

    def _end_round(self):
        curSuit = self.curBoard[0].suit_index
        maxValue = self.curBoard[0].value
        leadPlayerIndex = 0

        '''
        message = "[Simulator] curBoard: {}".format(self.curBoard)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        message = "[Simulator] curPlayer: {}".format(self.curPlayer)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        message = "[Simulator] curPlayers: {}".format(self.curPlayers)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        '''
        for i in range(4):
            card = self.curBoard[i]
            if card.suit_index == curSuit and card.value > maxValue:
                maxValue = card.value
                leadPlayerIndex = i

        '''
        message  = "[Simulator] leadPlayer : {}".format(self.curPlayers[leadPlayerIndex])
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        '''

        for i in range(4):
            card = self.curBoard[i]
            if card in self.game_score_cards:
                self.score_cards[self.curPlayers[leadPlayerIndex]].append(card)

        for i in range(4):
            picked_score_cards = self.score_cards[i]
            round_score = 0
            round_heart_score = 0
            is_double = False
            if len(picked_score_cards) > 0:
                for card in picked_score_cards:
                    if card == Card("QS"):
                        round_score -= 13
                    elif card == Card("TC"):
                        is_double = True
                    else:
                        round_heart_score -= 1
                if self.is_expose_card:
                    round_heart_score *= 2
                round_score += round_heart_score
                if is_double:
                    round_score *= 2
            self.scores[i] = round_score

        self.nextFirst = self.curPlayers[leadPlayerIndex]
        self.curBoard = []
        self.curPlayers = []

    # TODO need add club 10 and expose card feature
    def _others_pick_card(self, cards):
        # this candidate_cards is list<Card> not list<String> in montecarlobot.py
        candidate_cards = self._get_candidate_cards(cards)
        card_index = 0
        if not self.curBoard:
            for i in range(len(candidate_cards)):
                if candidate_cards[card_index].value > candidate_cards[i].value:
                    card_index = i 
        else:
            suit = self.curBoard[0].suit_index
            maxNum = self.curBoard[0].value
            for i in range(len(self.curBoard)):
                if self.curBoard[i].suit_index == suit and self.curBoard[i].value > maxNum:
                    maxNum = self.curBoard[i].value
            for i in range(len(candidate_cards)):
                cur_suit = candidate_cards[i].suit_index
                cur_num = candidate_cards[i].value
                prev_suit = candidate_cards[card_index].suit_index
                prev_num = candidate_cards[card_index].value
                if prev_suit == cur_suit:
                    if cur_suit == suit:
                        if cur_num < maxNum: 
                            if prev_num > maxNum or prev_num < cur_num:
                                card_index = i
                        elif cur_num > maxNum and prev_num > maxNum and len(self.curBoard) == 3:
                            if cur_num > prev_num:
                                card_index = i
                        elif cur_num < prev_num:
                            card_index = i
                        else:
                            pass
                    else:
                        if cur_num > prev_num:
                            card_index = i;
                else:
                    if cur_suit == 0 and cur_num == 12:
                        card_index = i;
                    elif prev_suit == 0 and prev_num == 12:
                        pass
                    elif cur_suit == 2:
                        card_index = i;
                    elif prev_suit == 2:
                        pass
                    elif cur_num > prev_num:
                        card_index = i;
                    else:
                        pass
            
        return candidate_cards[card_index]
