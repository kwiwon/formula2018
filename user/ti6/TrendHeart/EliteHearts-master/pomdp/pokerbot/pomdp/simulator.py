
import functools
import copy
from pokerbot.card import Card, SCORE_CARDS


class Simulator:

    def __init__(self):
        self.hand_cards_of_players = [[] for i in range(4)]
        self.cur_board = []
        self.heart_broken = False
        self.is_card_exposed = False
        self.scores = []
        self.score_cards = [[] for i in range(4)]
        self.cur_player_id = 0
        self.turn_end_player_id = []
        self.my_id = 0
        self.next_first = -1
        self.game_score_cards = SCORE_CARDS
        self.observation_buffer = [] # list of tuple(playerID, Card)

    def init_with_state(self, state):
        self.hand_cards_of_players = [[] for i in range(4)]
        for i in range(4):
            self.hand_cards_of_players[i].extend(state.players[i]["hasCards"])
        self.cur_board = []
        self.cur_board.extend(state.cur_board)
        self.heart_broken = state.heart_broken
        self.is_card_exposed = state.is_card_exposed
        self.scores = []
        for i in range(4):
            self.scores.append(state.players[i]["score"])
        self.score_cards = [[] for i in range(4)]
        for i in range(4):
            self.score_cards[i].extend(state.players[i]["scoreCards"])

        self.my_id = state.self_index
        
        self.cur_player_id = self.my_id

        self.turn_end_player_id = []
        for player_name, card in state.picked_cards_this_round:
            for player_id in range(4):
                if player_name == state.players[player_id]["playerName"]:
                    self.turn_end_player_id.append(player_id)

        self.observation_buffer = []

    '''
    State
  
        cur_board = [ Card("2D"), Card("5D") ]

        heart_broken = False
        is_card_exposed = False
        self_index = 0
        players = [
            {
                "playerName" : "player1"
                "hasCards" = [ Card("AH"), Card("9S") ]
                "lackCards" : set("D", "C")
                "self": False
                "scoreCards": [ Card("6H"), Card("TC")]
                "score" : 1 
            },
            {
                "playerName" : "player2"
                "hasCards" = [ ]
                "lackCards" : set("H")
                "self": False
                "scoreCards": [ Card("2H") ]
                "score" : 1 
            }, ...
        ]

        picked_cards = [(5, 'me', Card('2D')), (5, 'player2', Card('5D') ]
        # list of tuple(runoundNamer, playerName, Card)
        
        picked_cards_this_round = []
        # list of tuple(playerName, Card)
    '''
    def run(self, state):
        self.init_with_state(state)

        self._rollout()

        moon_shooter_id = -1
        for i in range(4):
            if len(self.score_cards[i]) == 15:
                moon_shooter_id = i
            elif len(self.score_cards[i]) == 14 and Card("TC") not in self.score_cards[i]:
                moon_shooter_id = i

        if moon_shooter_id != -1:
            if moon_shooter_id == self.my_id:
                return (self.scores[moon_shooter_id] * -1) * 4
            else:
                return (self.scores[moon_shooter_id]) * 4

        return self.scores[self.my_id]

    def step(self, state, card_to_play):
        assert isinstance(card_to_play, Card), "step, card_to_play is not type Card"
        self.init_with_state(state)
        ori_score = self.scores[self.my_id]

        self._play(self.cur_player_id, card_to_play)
        self.cur_player_id = (self.cur_player_id +1) %4

        while len(self.cur_board) < 4:
            card_to_play = self._decide(self.cur_player_id)
            self._play(self.cur_player_id, card_to_play)
            self.cur_player_id = (self.cur_player_id +1) %4
        self._end_round()

        self.cur_player_id = self.next_first
        while len(self.hand_cards_of_players[self.cur_player_id]) > 0 and self.cur_player_id != self.my_id:
            card_to_play = self._decide(self.cur_player_id)
            self._play(self.cur_player_id, card_to_play)
            self.cur_player_id = (self.cur_player_id +1) %4

        output_score = ori_score - self.scores[self.my_id]
        moon_shooter_id = -1
        for i in range(4):
            if len(self.score_cards[i]) == 15:
                moon_shooter_id = i
            elif len(self.score_cards[i]) == 14 and Card("TC") not in self.score_cards[i]:
                moon_shooter_id = i

        if moon_shooter_id != -1:
            if moon_shooter_id == self.my_id:
                output_score += (self.scores[moon_shooter_id] * -1) * 4
            else:
                output_score += (self.scores[moon_shooter_id]) * 4

        return_state = copy.deepcopy(state)
        return_state.cur_board = []
        return_state.cur_board.extend(self.cur_board)
        return_state.heart_broken = self.heart_broken
        for i in range(4):
            return_state.players[i]["hasCards"] = self.hand_cards_of_players[i].copy()
            return_state.players[i]["scoreCards"] = []
            return_state.players[i]["scoreCards"].extend(self.score_cards[i])
            return_state.players[i]["score"] = self.scores[i]

        return_state.picked_cards_this_round = []
        for i in range(len(self.turn_end_player_id)):
            for player_id in range(4):
                if self.turn_end_player_id[i] == player_id:
                    return_state.picked_cards_this_round.append((return_state.players[player_id]["playerName"], self.cur_board[i]))
        # list of tuple(playerName, Card)
        return_observation_buffer = [(state.players[player_id]["playerName"], card) for player_id, card in self.observation_buffer]

        return return_state, return_observation_buffer, output_score

    def _play(self, player_id, card):
        self.turn_end_player_id.append(player_id)
        self.cur_board.append(card)
        # list of tuple(playerName, Card)
        assert isinstance(card, Card), "_play card is not type Card"
        self.observation_buffer.append((player_id, card))

        if card.suit == "H":
            self.heart_broken = True

        self.hand_cards_of_players[player_id].remove(card)

    def _rollout(self):
        while len(self.cur_board) < 4:
            card_to_play = self._decide(self.cur_player_id)
            self._play(self.cur_player_id, card_to_play)
            self.cur_player_id = (self.cur_player_id +1) %4
        self._end_round()

        while len(self.hand_cards_of_players[self.next_first]) > 0:
            self.cur_player_id = self.next_first
            while len(self.cur_board) < 4:
                card_to_play = self._decide(self.cur_player_id)
                self._play(self.cur_player_id, card_to_play)
                self.cur_player_id = (self.cur_player_id +1) %4
            self._end_round()

    def _end_round(self):
        #print("round end, cur board: {}".format(self.cur_board))
        #print("round end, turn_end_player_id: {}".format(self.turn_end_player_id))

        lead_card = self.cur_board[0]
        lead_player_id = self.turn_end_player_id[0]
        for i in range(4):
            card = self.cur_board[i]
            if card.suit == lead_card.suit and card.value > lead_card.value:
                lead_card = card
                lead_player_id = self.turn_end_player_id[i]

        for card in self.cur_board:
            if card in self.game_score_cards:
                self.score_cards[lead_player_id].append(card)

        for player_id in range(4):
            picked_score_cards = self.score_cards[player_id]
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
                if self.is_card_exposed:
                    round_heart_score *= 2
                round_score += round_heart_score
                if is_double:
                    round_score *= 2
            self.scores[player_id] = round_score

        #for i in range(4):
        #    print("round end, player {} score card: {}".format(i, self.score_cards[i]))
        #print("round end, scores: {}".format(self.scores))

        self.next_first = lead_player_id
        self.cur_board = []
        self.turn_end_player_id = []

    def _get_candidate_cards(self, cards):
        candidate_cards = []
        if not self.cur_board:
            if Card('2C') in cards:
                candidate_cards.append('2C')
            elif self.heart_broken:
                candidate_cards.extend(cards)
            else:
                candidate_cards = list(filter(lambda c: c.suit != "H", cards))
        else:
            lead_card = self.cur_board[0]
            candidate_cards = list(filter(lambda c: c.suit == lead_card.suit, cards))
        
        if len(candidate_cards) == 0:
            candidate_cards.extend(cards)

        return candidate_cards

    def _decide(self, player_id):
        # this candidate_cards is list<Card>
        candidate_cards = self._get_candidate_cards(self.hand_cards_of_players[player_id])
        choosen_card = candidate_cards[0]
        if not self.cur_board:
            choosen_card = functools.reduce(lambda c1,c2: c1 if c1.value < c2.value else c2, candidate_cards)
        else:
            lead_card = functools.reduce(lambda c1, c2: c1 if c1.suit == c2.suit and c1.value > c2.value else c2, self.cur_board)
            for card in candidate_cards:
                if card.suit == choosen_card.suit:
                    if card.suit == lead_card.suit:
                        if card.value < lead_card.value:
                            if choosen_card.value > lead_card.value or choosen_card.value < card.value:
                                choosen_card = card
                        elif choosen_card.value > lead_card.value and len(self.cur_board) == 3:
                            if card.value > choosen_card.value:
                                choosen_card = card
                        elif choosen_card.value > card.value:
                            choosen_card = card
                    else:
                        if card.value > choosen_card.value or card == Card('QS'):
                            choosen_card = card
                else:
                    if card == Card("QS"):
                        choosen_card = card
                    elif choosen_card == Card("QS"):
                        pass
                    elif card.suit == "H" and choosen_card.suit != "H":
                        choosen_card = card
                    elif card.suit != "H" and choosen_card.suit == "H":
                        pass
                    elif card.value > choosen_card.value:
                        choosen_card = card
        assert isinstance(choosen_card, Card), "choosen_card is not type Card {} {}".format(candidate_cards, map(type, candidate_cards))
        return choosen_card
