from websocket import create_connection
import json


class Connector:
    def __init__(self, player_name, player_number, token, connect_url, agent, local=False):
        self._player_name = player_name
        self._player_number = player_number
        self._token = token
        self._connect_url = connect_url
        self._agent = agent
        self._ws = None
        self._local = local
        self._event_to_action = {
            'new_game': self._on_new_game,
            'new_deal': self._on_new_deal,
            'new_round': self._on_new_round,
            'pass_cards': self._on_pass_cards,
            'receive_opponent_cards': self._on_receive_opponent_cards,
            'expose_cards': self._on_expose_cards,
            'expose_cards_end': self._on_expose_cards_end,
            'your_turn': self._on_your_turn,
            'turn_end': self._on_turn_end,
            'round_end': self._on_round_end,
            'deal_end': self._on_deal_end,
            'game_end': self._on_game_end
        }

    # actions for each events
    def _on_new_game(self, data):
        self._agent.new_game(data)

    def _on_new_deal(self, data):
        self._agent.new_deal(data)

    def _on_new_round(self, data):
        self._agent.new_round(data)

    def _on_pass_cards(self, data):
        cards = self._agent.pass_cards(data)
        message = json.dumps({
            'eventName': 'pass_my_cards',
            'data': {
                'dealNumber': data['dealNumber'],
                'cards': cards
            }
        })
        if self._local:
            return message
        self._ws.send(message)

    def _on_receive_opponent_cards(self, data):
        self._agent.receive_opponent_cards(data)

    def _on_expose_cards(self, data):
        is_exposed = self._agent.expose_cards(data)
        if is_exposed:
            cards = ['AH']
        else:
            cards = []
        message = json.dumps({
                'eventName': 'expose_my_cards',
                'data': {
                    'dealNumber': data['dealNumber'],
                    'cards': cards
                }
            })
        if self._local:
            return message

        self._ws.send(message)

    def _on_expose_cards_end(self, data):
        self._agent.expose_cards_end(data)

    def _on_your_turn(self, data):
        card = self._agent.pick_card(data)
        message = json.dumps({
            'eventName': 'pick_card',
            'data': {
                'dealNumber': data['dealNumber'],
                'roundNumber': data['roundNumber'],
                'turnCard': card
            }
        })
        if self._local:
            return message
        self._ws.send(message)

    def _on_turn_end(self, data):
        self._agent.turn_end(data)

    def _on_round_end(self, data):
        self._agent.round_end(data)

    def _on_deal_end(self, data):
        self._agent.deal_end(data)
        self._agent.reset()

    def _on_game_end(self, data):
        self._agent.game_end(data)
        if self._ws:
            self._ws.close()
            self._ws = None

    def _take_action(self, action, data):
        if action in self._event_to_action:
            return self._event_to_action[action](data)

    def take_action(self, data):
        msg = json.loads(data)
        event_name = msg['eventName']
        data = msg['data']
        return self._take_action(event_name, data)

    def listen(self):
        # try:
            self._ws = create_connection(self._connect_url)
            self._ws.send(json.dumps({
                'eventName': 'join',
                'data': {
                    'playerNumber': self._player_number,
                    'playerName': self._player_name,
                    'token': self._token
                }
            }))

            while True:
                response = self._ws.recv()
                msg = json.loads(response)
                event_name = msg['eventName']
                data = msg['data']
                self._take_action(event_name, data)
        # except:
        #     self.listen()