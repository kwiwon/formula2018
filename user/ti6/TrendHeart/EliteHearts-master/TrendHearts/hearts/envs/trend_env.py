import json
from hearts.envs.utils import Deck
from hearts.envs.utils import Card
import random
from pathlib import Path
import csv
import datetime


class DealState:
    def __init__(self):
        self.cards = set()
        self.initial_card = set()
        self.round_card = None
        self.passing_cards = set()
        self.round_player = None
        self.receive_opponent_cards = set()
        self.received_from = ''
        self.deal_score = 0
        self.score_cards = set()
        self.is_shooting_the_moon = False
        self.expose_AH = False
        self.has_AH = False


class Agent:
    def __init__(self, player_name, player_number, token, agent):
        self.player_name = player_name
        self.player_number = player_number
        self.token = token
        self.agent = agent
        self.game_score = 0
        self.rank = 0
        self.deal_state = [DealState(), DealState(), DealState(), DealState()]
        self.server_random = False
        self.error_count = 0


class TrendHeartsEnv:
    def __init__(self, render=False, log=True):
        self._deck = Deck()
        self._agent_list = []
        self._passing_card_order = [1, -1, 2, 0]
        self._current_round_num = 1
        self._current_deal_num = 1
        self._current_turn_num = 1
        self._is_heart_exposed = False
        self._seat_position = [3, 0, 1, 2]  # seat from 0 - 4 is north-east-south-west seat, values are agent index
        self._current_round_winner = None
        self._round_player_order = None
        self._is_heart_broken = False
        self._current_suit = ''
        self._render = render
        self._current_game_num = 1
        self._log_writer = None

        # initialise log
        if log:
            log_dir_path = Path('log')
            try:
                log_dir_path.mkdir()
            except FileExistsError:
                pass
            log_path = log_dir_path / (datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.csv')
            self._log_writer = csv.writer(open(log_path, 'w', newline=''))

    def join(self, agent, message):
        # emulate the join event
        message = json.loads(message)
        event = message['eventName']
        data = message['data']
        if event == 'join':
            _agent = Agent(
                player_name=data['playerName'],
                player_number=data['playerNumber'],
                token=data['token'],
                agent=agent
            )
            self._agent_list.append(_agent)

    def reset_agent(self):
        for agent in self._agent_list:
            agent.game_score = 0
            agent.rank = 0
            agent.deal_state = [DealState(), DealState(), DealState(), DealState()]
            agent.server_random = False
            agent.error_count = 0

    def reset(self):
        # reset agent
        self.reset_agent()

        # reset environment
        self._deck = Deck()
        self._current_round_num = 1
        self._current_deal_num = 1
        self._current_turn_num = 1
        self._is_heart_exposed = False
        self._current_round_winner = None
        self._round_player_order = None
        self._is_heart_broken = False
        self._current_suit = ''

    def new_game(self):
        # sent new game event to all agent
        message = {'eventName': 'new_game', 'data': {}}
        players = []
        info_list = ['status']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num, info_list))
        message['data']['players'] = players

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def build_agent_info(self, agent, deal_number, info_list):
        # default build player name and player number
        player_info = {'playerNumber': agent.player_number, 'playerName': agent.player_name}

        if 'gameScore' in info_list:
            player_info['gameScore'] = agent.game_score

        if 'errorCount' in info_list:
            player_info['errorCount'] = agent.error_count

        if 'timeoutCount' in info_list:
            player_info['timeoutCount'] = 0

        if 'dealScore' in info_list:
            player_info['dealScore'] = agent.deal_state[deal_number].deal_score

        if 'scoreCards' in info_list:
            player_info['scoreCards'] = []
            for card in agent.deal_state[deal_number].score_cards:
                player_info['scoreCards'].append(card.to_string())

        if 'pickedCards' in info_list:
            player_info['pickedCards'] = []
            for card in agent.deal_state[deal_number].passing_cards:
                player_info['pickedCards'].append(card.to_string())

        if 'receivedCards' in info_list:
            player_info['receivedCards'] = []
            for card in agent.deal_state[deal_number].receive_opponent_cards:
                player_info['receivedCards'].append(card.to_string())

        if 'receivedFrom' in info_list:
            player_info['receivedFrom'] = agent.deal_state[deal_number].received_from

        if 'exposedCards' in info_list:
            if agent.deal_state[deal_number].expose_AH:
                player_info['exposedCards'] = ['AH']
            else:
                player_info['exposedCards'] = []

        if 'shootingTheMoon' in info_list:
            player_info['shootingTheMoon'] = agent.deal_state[deal_number].is_shooting_the_moon

        if 'status' in info_list:
            player_info['status'] = 0

        if 'cards' in info_list:
            player_info['cards'] = []
            for card in agent.deal_state[deal_number].cards:
                player_info['cards'].append(card.to_string())

        if 'cardsCount' in info_list:
            player_info['cardsCount'] = len(agent.deal_state[deal_number].cards)

        if 'serverRandom' in info_list:
            player_info['serverRandom'] = agent.server_random

        if 'roundCard' in info_list:
            if agent.deal_state[deal_number].round_card is not None:
                player_info['roundCard'] = agent.deal_state[deal_number].round_card.to_string()

        if 'initialCards' in info_list:
            player_info['initialCards'] = []
            for card in agent.deal_state[deal_number].initial_card:
                player_info['initialCards'].append(card.to_string())

        if 'rank' in info_list:
            player_info['rank'] = agent.rank

        if 'deals' in info_list:
            player_info['deals'] = []
            for deal_num, deal_state in enumerate(agent.deal_state):
                deal_info = {'dealNumber': deal_num + 1, 'score': deal_state.deal_score}
                if deal_state.expose_AH:
                    deal_info['exposedCards'] = ['AH']
                else:
                    deal_info['exposedCards'] = []
                    player_info['deals'].append(deal_info)

        return player_info

    def new_deal(self):
        # new a deck
        self._deck = Deck()

        # deal cards to each agent
        self._deal_cards()

        # send new deal event to agent
        message = {'eventName': 'new_deal', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]

            # add self information
            self_info = self.build_agent_info(agent, self._current_deal_num - 1, info_list)

            message['data']['self'] = self_info
            message['data']['players'] = players
            agent.agent.take_action(json.dumps(message))

    def _deal_cards(self):
        i = 0
        while self._deck.size > 0:
            card = self._deck.draw()
            self._agent_list[i % 4].deal_state[self._current_deal_num - 1].cards.add(card)
            self._agent_list[i % 4].deal_state[self._current_deal_num - 1].initial_card.add(card)
            i += 1

    def pass_cards(self):
        message = {'eventName': 'pass_cards', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        for seat, agent_position in enumerate(self._seat_position):
            agent = self._agent_list[agent_position]

            # add self information
            self_info = self.build_agent_info(agent, self._current_deal_num - 1, info_list)

            message['data']['self'] = self_info
            message['data']['players'] = players

            # add receiver information
            receiver_seat = (seat + self._passing_card_order[self._current_deal_num - 1]) % 4
            if receiver_seat < 0:
                receiver_seat = 3
            receiver = self._agent_list[self._seat_position[receiver_seat]]
            message['data']['receiver'] = receiver.player_name

            # expected to receive valid pass my card event from agent
            response = agent.agent.take_action(json.dumps(message))

            # record pass cards
            response = json.loads(response)
            receiver.deal_state[self._current_deal_num - 1].received_from = agent.player_name
            for card_str in response['data']['cards']:
                rank, suit = card_str
                suit = suit.lower()
                rank = rank.upper()
                rank_suit = rank + suit
                card = Card(rank_suit)
                agent.deal_state[self._current_deal_num - 1].passing_cards.add(card)
                receiver.deal_state[self._current_deal_num - 1].receive_opponent_cards.add(card)

    def receive_opponent_cards(self):
        message = {'eventName': 'receive_opponent_cards', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'receivedCards', 'receivedFrom']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]

            # add self information
            self_info = self.build_agent_info(agent, self._current_deal_num - 1, info_list)

            message['data']['self'] = self_info
            message['data']['players'] = players

            agent.agent.take_action(json.dumps(message))

    def pass_cards_end(self):
        message = {'eventName': 'pass_cards_end', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'pickedCards', 'receivedCards', 'receivedFrom']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def resolve_passing_cards(self):
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            # remove passing cards
            for card in agent.deal_state[self._current_deal_num - 1].passing_cards:
                agent.deal_state[self._current_deal_num - 1].cards.remove(card)

            # add get cards
            for card in agent.deal_state[self._current_deal_num - 1].receive_opponent_cards:
                agent.deal_state[self._current_deal_num - 1].cards.add(card)

    # select AH owner
    def get_ah_owner(self):
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            if Card('Ah') in agent.deal_state[self._current_deal_num - 1].cards:
                agent.deal_state[self._current_deal_num - 1].has_AH = True
                return agent

    def expose_cards(self):
        message = {'eventName': 'expose_cards', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        # ask AH owner to expose or not
        agent = self.get_ah_owner()

        # add self information
        self_info = self.build_agent_info(agent, self._current_deal_num - 1, info_list)

        # add candidate card
        self_info['candidateCards'] = ['AH']

        message['data']['self'] = self_info
        message['data']['players'] = players

        # expect to get response from agent
        response = agent.agent.take_action(json.dumps(message))
        response = json.loads(response)
        if response['data']['cards']:
            self._is_heart_exposed = True
            agent.deal_state[self._current_deal_num - 1].expose_AH = True

    def expose_cards_end(self):
        message = {'eventName': 'expose_cards_end', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'pickedCards', 'receivedCards', 'receivedFrom',
                     'exposedCards']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def find_round_one_player_seat_position(self):
        for seat, agent_position in enumerate(self._seat_position):
            agent = self._agent_list[agent_position]
            if Card('2c') in agent.deal_state[self._current_deal_num - 1].cards:
                self._current_round_winner = seat
                break

    def update_round_players_order(self):
        self._round_player_order = []
        for i in range(4):
            self._round_player_order.append((self._current_round_winner + i) % 4)

    def build_round_player_message(self):
        round_players_message = []

        for seat in self._round_player_order:
            agent_position = self._seat_position[seat]
            agent = self._agent_list[agent_position]
            round_players_message.append(agent.player_name)
        return round_players_message

    def new_round(self):
        message = {'eventName': 'new_round', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num
        message['data']['roundNumber'] = self._current_round_num

        # player information
        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'scoreCards', 'exposedCards']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players
        message['data']['roundPlayers'] = self.build_round_player_message()

        # send event to all agents
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def your_turn(self):
        message = {'eventName': 'your_turn', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num
        message['data']['roundNumber'] = self._current_round_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'scoreCards', 'roundCard', 'serverRandom',
                     'exposedCards']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        # add self information
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'exposedCards']
        seat = self._round_player_order[self._current_turn_num - 1]
        agent_position = self._seat_position[seat]
        agent = self._agent_list[agent_position]
        self_info = self.build_agent_info(agent, self._current_deal_num - 1, info_list)

        # build candidate cards
        candidate_cards = []

        # for round 1
        if self._current_round_num == 1:
            # first turn, only 2c can be played
            if self._current_turn_num == 1:
                candidate_cards.append('2C')
            else:
                # must follow the suit
                for card in agent.deal_state[self._current_deal_num - 1].cards:
                    if card.suit.upper() == self._current_suit:
                        candidate_cards.append(card.to_string())
                # cannot follow the suit
                if len(candidate_cards) == 0:
                    # for other turns, can't play hearts and QS for first round
                    for card in agent.deal_state[self._current_deal_num - 1].cards:
                        if card.suit.upper() != 'H' and card != Card('Qs'):
                            candidate_cards.append(card.to_string())
                    if len(candidate_cards) == 0:
                        # all penalty cards in hands, can't play Qs only
                        for card in agent.deal_state[self._current_deal_num - 1].cards:
                            if card != Card('Qs'):
                                candidate_cards.append(card.to_string())
        # not first round
        else:
            # for the round winner
            if self._current_turn_num == 1:
                if self._is_heart_broken is True:
                    # can play any cards
                    for card in agent.deal_state[self._current_deal_num - 1].cards:
                        candidate_cards.append(card.to_string())
                else:
                    # can play any cards but the not the heart
                    for card in agent.deal_state[self._current_deal_num - 1].cards:
                        if card.suit.upper() != 'H':
                            candidate_cards.append(card.to_string())

                    # if only hearts left, all cards can be played
                    if len(candidate_cards) == 0:
                        for card in agent.deal_state[self._current_deal_num - 1].cards:
                            candidate_cards.append(card.to_string())
            else:
                # must follow the suit
                for card in agent.deal_state[self._current_deal_num - 1].cards:
                    if card.suit.upper() == self._current_suit:
                        candidate_cards.append(card.to_string())

                    if len(candidate_cards) == 0:
                        # cannot follow the suit
                        if self._is_heart_broken is True:
                            # hearts has broken, can play any cards
                            for card in agent.deal_state[self._current_deal_num - 1].cards:
                                candidate_cards.append(card.to_string())
                        else:
                            # hearts hasn't broken, should play other cards
                            for card in agent.deal_state[self._current_deal_num - 1].cards:
                                if card.suit.upper() != 'H':
                                    candidate_cards.append(card.to_string())

                            # has only heart cards
                            if len(candidate_cards) == 0:
                                for card in agent.deal_state[self._current_deal_num - 1].cards:
                                    candidate_cards.append(card.to_string())

        self_info['candidateCards'] = candidate_cards
        message['data']['self'] = self_info
        message['data']['players'] = players
        message['data']['roundPlayers'] = self.build_round_player_message()

        # expected to get pick card event
        response = agent.agent.take_action(json.dumps(message))
        response = json.loads(response)
        card_str = response['data']['turnCard']

        if card_str not in candidate_cards:
            card_str = random.choice(candidate_cards)
            agent.server_random = True
            agent.error_count += 1
        else:
            agent.server_random = False

        rank, suit = card_str
        suit = suit.lower()
        rank = rank.upper()
        rank_suit = rank + suit
        card = Card(rank_suit)

        # remove cards from agent hand
        agent.deal_state[self._current_deal_num - 1].cards.remove(card)

        # add cards to agent round cards
        agent.deal_state[self._current_deal_num - 1].round_card = card

        # set suit if is first turn
        if self._current_turn_num == 1:
            self._current_suit = card.suit.upper()

        # broken hearts
        if card.suit.upper() == 'H':
            self._is_heart_broken = True

    def turn_end(self):
        message = {'eventName': 'turn_end', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num
        message['data']['roundNumber'] = self._current_round_num
        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'scoreCards', 'roundCard', 'serverRandom',
                     'exposedCards']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        seat = self._round_player_order[self._current_turn_num - 1]
        agent_position = self._seat_position[seat]
        turn_agent = self._agent_list[agent_position]

        message['data']['players'] = players
        message['data']['roundPlayers'] = self.build_round_player_message()
        message['data']['turnPlayer'] = turn_agent.player_name
        message['data']['turnCard'] = turn_agent.deal_state[self._current_deal_num - 1].round_card.to_string()
        message['serverRandom'] = turn_agent.server_random

        # send event to all agents
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def resolve_round_winner(self):
        round_winner_position = None
        round_winner = None
        largest_card = None
        for seat, agent_position in enumerate(self._seat_position):
            agent = self._agent_list[agent_position]
            # match suit only
            if agent.deal_state[self._current_deal_num - 1].round_card.suit.upper() == self._current_suit:
                if largest_card is None or agent.deal_state[self._current_deal_num - 1].round_card > largest_card:
                    largest_card = agent.deal_state[self._current_deal_num - 1].round_card
                    round_winner_position = seat
                    round_winner = agent

        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            card = agent.deal_state[self._current_deal_num - 1].round_card
            if card.suit.upper() == 'H' or card == Card('Qs') or card == Card('Tc'):
                round_winner.deal_state[self._current_deal_num - 1].score_cards.add(card)
        self._current_round_winner = round_winner_position

    def calculate_deal_score(self):
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            deal_score = 0
            for card in agent.deal_state[self._current_deal_num - 1].score_cards:
                if card.suit.upper() == 'H':
                    deal_score += -1

            # double the hearts score if heart exposed
            if self._is_heart_exposed:
                deal_score *= 2
            if Card('Qs') in agent.deal_state[self._current_deal_num - 1].score_cards:
                deal_score += -13

            # check shooting the moon
            if deal_score == (-13 + -13) or deal_score == (-13 * 2 + -13):  # heart exposed or not
                agent.deal_state[self._current_deal_num - 1].is_shooting_the_moon = True
                # make score positive
                deal_score *= -1
                # multiply by 4
                deal_score *= 4

            if Card('Tc') in agent.deal_state[self._current_deal_num - 1].score_cards:
                deal_score *= 2
            agent.deal_state[self._current_deal_num - 1].deal_score = deal_score

    def round_end(self):
        message = {'eventName': 'round_end', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num
        message['data']['roundNumber'] = self._current_round_num

        players = []
        info_list = ['gameScore', 'dealScore', 'cards', 'cardsCount', 'scoreCards', 'roundCard', 'serverRandom',
                     'exposedCards']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players
        message['data']['roundPlayers'] = self.build_round_player_message()

        round_winner_position = self._seat_position[self._current_round_winner]
        round_winner = self._agent_list[round_winner_position]
        message['roundPlayer'] = round_winner.player_name

        # send event to all agents
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def clear_agent_round_card(self, deal_number):
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.deal_state[deal_number].round_card = None

    def deal_end(self):
        message = {'eventName': 'deal_end', 'data': {}}
        message['data']['dealNumber'] = self._current_deal_num
        message['data']['roundNumber'] = self._current_round_num
        players = []
        info_list = ['gameScore', 'errorCount', 'timeoutCount', 'dealScore', 'scoreCards', 'pickedCards',
                     'receivedCards', 'receivedFrom', 'exposedCards', 'shootingTheMoon', 'initialCards', 'status']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players

        # send event to all agents
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def resolve_game_score(self):
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.game_score += agent.deal_state[self._current_deal_num - 1].deal_score

    def resolve_rank(self):
        ranked_agent_list = sorted(self._agent_list, key=lambda x: x.game_score, reverse=True)
        for i, agent in enumerate(ranked_agent_list):
            agent.rank = i + 1

    def game_end(self):
        message = {'eventName': 'game_end', 'data': {}}
        players = []
        info_list = ['gameScore', 'timeoutCount', 'errorCount', 'rank', 'deals']
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            players.append(self.build_agent_info(agent, self._current_deal_num - 1, info_list))

        message['data']['players'] = players

        # send event to all agents
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            agent.agent.take_action(json.dumps(message))

    def render(self):
        print(f'deal{self._current_deal_num} round{self._current_round_num}')
        # print agent hand cards
        for seat, agent_position in enumerate(self._seat_position):
            agent = self._agent_list[agent_position]
            print(f'{agent.player_name} ', end='')
            for card in agent.deal_state[self._current_deal_num - 1].cards:
                print(card, end='')
            if seat == self._current_round_winner:
                print(' *  ', end='')
            else:
                print('    ', end='')
            print(f'deal score:{agent.deal_state[self._current_deal_num - 1].deal_score} game score:{agent.game_score}',
                  end='')
            print()

        # print round cards
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            print(agent.deal_state[self._current_deal_num - 1].round_card, end='')
        print()
        print()

    def write_log(self):
        # write header
        if self._current_game_num == 1 and self._current_deal_num == 1 and self._current_round_num == 1:
            header = ['game_num', 'deal_num', 'round_num']
            for agent_position in self._seat_position:
                agent = self._agent_list[agent_position]
                header.append(f'{agent.player_name}_hands')
            header.append('round_cards')
            for agent_position in self._seat_position:
                agent = self._agent_list[agent_position]
                header.append(f'{agent.player_name}_deal_score')
            for agent_position in self._seat_position:
                agent = self._agent_list[agent_position]
                header.append(f'{agent.player_name}_game_score')
            self._log_writer.writerow(header)

        content = []
        content.append(self._current_game_num)
        content.append(self._current_deal_num)
        content.append(self._current_round_num)
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            cards = [card.to_string() for card in agent.deal_state[self._current_deal_num - 1].cards]
            content.append(cards)

        round_card = []
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            round_card.append(agent.deal_state[self._current_deal_num - 1].round_card.to_string())
        content.append(round_card)
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            content.append(agent.deal_state[self._current_deal_num - 1].deal_score)
        for agent_position in self._seat_position:
            agent = self._agent_list[agent_position]
            content.append(agent.game_score)
        self._log_writer.writerow(content)

    def game_start(self):
        # game start
        self.new_game()

        # 4 deals
        while self._current_deal_num <= 4:
            # reset heart broken
            self._is_heart_broken = False
            self.new_deal()

            # pass card event
            if self._current_deal_num < 4:
                self.pass_cards()
                self.receive_opponent_cards()
                self.pass_cards_end()
                self.resolve_passing_cards()

            # expose cards
            self.expose_cards()
            self.expose_cards_end()

            # 13 rounds
            self.find_round_one_player_seat_position()
            while self._current_round_num <= 13:
                self.update_round_players_order()
                self.new_round()

                # 4 turns
                while self._current_turn_num <= 4:
                    self.your_turn()
                    self.turn_end()
                    self._current_turn_num += 1

                self.resolve_round_winner()
                self.calculate_deal_score()
                self.round_end()

                # render
                if self._render:
                    self.render()

                # write log
                if self._log_writer:
                    self.write_log()

                self.clear_agent_round_card(self._current_deal_num - 1)
                self._current_turn_num = 1
                self._current_round_num += 1

            # round number - 1 as it is now 14 when exit above loop
            self._current_round_num -= 1
            self.deal_end()
            self.resolve_game_score()
            self._current_round_num = 1
            self._current_deal_num += 1

        self.resolve_rank()
        self.game_end()
        self._current_game_num += 1
