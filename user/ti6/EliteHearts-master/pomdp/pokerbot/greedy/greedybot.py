from ..pokerbot import PokerBot
from ..pokerbot import Card


class GreedyBot(PokerBot):

    def __init__(self,name, system_log):
        super(GreedyBot,self).__init__(name, system_log)
        self.my_hand_cards=[]
        self.expose_card=False
        self.my_pass_card=[]

    def new_deal(self,data):
        super(GreedyBot, self).new_deal(data)
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
        self.update_pass_cards(data["dealNumber"], pass_cards)
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
        
        board=[]
        for player_name in data['roundPlayers']:
            if data['self']['playerName'] == player_name:
                break
            else:
                for player in data['players']:
                    if player['playerName'] == player_name:
                        board.append(Card(player['roundCard']))

        if len(board) > 0:
            suit=board[0].suit_index;
            maxNum=board[0].value
            for i in range(0, len(board)):
                if board[i].suit_index == suit and board[i].value > maxNum:
                    maxNum = board[i].value

            for i in range(0, len(candidate_cards)):
                cur_suit = Card(candidate_cards[i]).suit_index
                cur_num = Card(candidate_cards[i]).value
                prev_suit= Card(candidate_cards[card_index]).suit_index
                prev_num= Card(candidate_cards[card_index]).value
                if prev_suit == cur_suit:
                    if cur_suit == suit:
                        if cur_num < maxNum: 
                            if prev_num > maxNum or prev_num < cur_num:
                                card_index = i
                        elif cur_num > maxNum and prev_num > maxNum and len(board) == 3:
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
        else:
            for i in range(0, len(candidate_cards)):
                if Card(candidate_cards[card_index]).value > Card(candidate_cards[i]).value:
                    card_index=i
        
        message = "Pick Card:{}".format(candidate_cards[card_index])
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        return candidate_cards[card_index]

    def expose_my_cards(self,yourcards):
        expose_card=[]
        for card in self.my_hand_cards:
            if card==Card("AH"):
                expose_card.append(str(card))
        message = "Expose Cards:{}".format(expose_card)
        self.system_log.show_message(message)
        self.system_log.save_logs(message)
        return expose_card

    def expose_cards_end(self,data):
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
        super(GreedyBot, self).receive_opponent_cards(data)
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

    def round_end(self,data):
        try:
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
