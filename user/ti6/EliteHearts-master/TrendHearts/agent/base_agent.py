from agent.utils import Card
from agent.utils import pass_random_card


debug = False


class BaseAgent:
    def __init__(self, player_name):
        self._player_name = player_name
        self._penalty_card = {
            Card('QS'), Card('TS'), Card('2H'), Card('3H'), Card('4H'), Card('5H'), Card('6H'),
            Card('7H'), Card('8H'), Card('9H'), Card('TH'), Card('JH'), Card('QH'), Card('KH'),
            Card('AH')
        }
        self._current_deal = 0
        self._current_round = 0
        self._player_hand_cards = set()
        self._north_player_cards_history = set()
        self._east_player_cards_history = set()
        self._south_player_cards_history = set()
        self._west_player_cards_history = set()
        self._players_cards_history = [
            self._north_player_cards_history,
            self._east_player_cards_history,
            self._south_player_cards_history,
            self._west_player_cards_history
        ]
        self._player_round_card = [None] * 4
        self._player_name_to_position = {}
        self._player_game_scores = [0] * 4
        self._player_deal_scores = [0] * 4
        self._is_card_exposed = False

    def _resolve_position(self, data):
        for i, player in enumerate(data['players']):
            player_name = player['playerName']
            self._player_name_to_position[player_name] = i

    def _resolve_deal_score(self, data):
        for player in data['players']:
            player_name = player['playerName']
            player_deal_score = player['dealScore']
            if not self._player_name_to_position:
                self._resolve_position(data)
            player_position = self._player_name_to_position[player_name]
            self._player_deal_scores[player_position] = player_deal_score

    def _resolve_game_score(self, data):
        for player in data['players']:
            player_name = player['playerName']
            player_game_score = player['gameScore']
            player_position = self._player_name_to_position[player_name]
            self._player_game_scores[player_position] = player_game_score

    def new_peer(self, data):
        pass

    def new_game(self, data):
        # resolve all players' positions
        self._resolve_position(data)
        if debug:
            print('new_game')
            print(f'player_position: {self._player_name_to_position}')

    def new_round(self, data):
        self._current_round = data['roundNumber']
        self._player_round_card = [None] * 4
        if debug:
            print(f'round{self._current_round}')
            print()

    def new_deal(self, data):
        self._current_deal = data['dealNumber']

        # resolve deal scores
        self._resolve_deal_score(data)
        if debug:
            print(f'new_deal: deal{self._current_deal}')
            print(f'deal_scores: {self._player_deal_scores}')

        # resolve game scores
        self._resolve_game_score(data)
        if debug:
            print(f'game_scores: {self._player_game_scores}')

        # set cards
        for card in data['self']['cards']:
            self._player_hand_cards.add(Card(card))
        if debug:
            print(f'my hands: {self._player_hand_cards}')
            print()

    def pass_cards(self, data):
        cards = pass_random_card(data)
        if debug:
            print('pass_card')
            print(f'cards: {cards}')
            print()
        return cards

    def expose_cards(self, data):
        is_expose = False
        if debug:
            print('expose_cards')
            print(f'is_exposed: {is_expose}')
            print()
        return is_expose

    def receive_opponent_cards(self, data):
        for card_str in data['self']['receivedCards']:
            self._player_hand_cards.add(Card(card_str))

    def pass_cards_end(self, data):
        pass

    def expose_cards_end(self, data):
        exposed_player = None
        for player in data['players']:
            if 'exposedCards' in player and player['exposedCards'] and len(player['exposedCards']) > 0:
                exposed_player = player['playerName']
        if debug:
            print('expose_cards_end')
            print(f'{exposed_player} exposed card')
            print()

    def pick_card(self, data):
        pass

    def turn_end(self, data):
        player_name = data['turnPlayer']
        card = Card(data['turnCard'])
        if not self._player_name_to_position:
            self._resolve_position(data)
        player_position = self._player_name_to_position[player_name]
        player_card = self._players_cards_history[player_position]
        player_card.add(card)
        self._player_round_card[player_position] = card
        if debug:
            print('turn_end')
            print(f'{player_name} pick {card}')
            print(f'{player_name} history: {player_card}')
            print()

    def round_end(self, data):
        # resolve deal score
        self._resolve_deal_score(data)
        if debug:
            print('round_end')
            print(f'deal_scores: {self._player_deal_scores}')
            print()

    def deal_end(self, data):
        # resolve deal score
        self._resolve_deal_score(data)
        if debug:
            print('deal_end')
            print(f'deal_scores: {self._player_deal_scores}')
            print()

    def game_end(self, data):
        # resolve game score
        self._resolve_game_score(data)
        if debug:
            print('game_end')
            print(f'game_scores: {self._player_game_scores}')
            print()

    def reset(self):
        for card_history in self._players_cards_history:
            card_history.clear()
        self._player_hand_cards.clear()
        self._is_card_exposed = False