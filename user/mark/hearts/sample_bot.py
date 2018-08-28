# coding=UTF-8
from websocket import create_connection
from bot.LowPlayBot import LowPlayBot

import json
import sys

class PokerSocket(object):
    def __init__(self, player_name, player_number, token, connect_url, poker_bot):
        self.player_name = player_name
        self.connect_url = connect_url
        self.player_number = player_number
        self.poker_bot = poker_bot
        self.token = token

    def takeAction(self, action, data):

        print 'Action :', action

        if action == "new_deal":
            self.poker_bot.receive_cards(data)

        elif action == "pass_cards":
            pass_cards = self.poker_bot.pass_cards(data)
            self.ws.send(json.dumps(
                {
                    "eventName": "pass_my_cards",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "cards": pass_cards
                    }
                }))

        elif action == "receive_opponent_cards":
            self.poker_bot.receive_opponent_cards(data)

        elif action == "expose_cards":
            export_cards = self.poker_bot.expose_my_cards(data)
            if export_cards is not None:
                self.ws.send(json.dumps(
                    {
                        "eventName": "expose_my_cards",
                        "data": {
                            "dealNumber": data['dealNumber'],
                            "cards": export_cards
                        }
                    }))

        elif action == "expose_cards_end":
            self.poker_bot.expose_cards_end(data)

        elif action == "your_turn":
            pick_card = self.poker_bot.pick_card(data)

            message = "Send message:{}".format(json.dumps(
                {
                    "eventName": "pick_card",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "roundNumber": data['roundNumber'],
                        "turnCard": pick_card
                    }
                }))

            self.ws.send(json.dumps(
                {
                    "eventName": "pick_card",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "roundNumber": data['roundNumber'],
                        "turnCard": pick_card
                    }
                }))

        elif action == "turn_end":
            self.poker_bot.turn_end(data)

        elif action == "round_end":
            self.poker_bot.round_end(data)

        elif action == "deal_end":
            self.poker_bot.deal_end(data)
            self.poker_bot.reset_card_his()

        elif action == "game_end":
            self.poker_bot.game_over(data)
            self.ws.send(json.dumps({
                "eventName": "stop_game",
                "data": {}
            }))
            self.ws.close()

    def doListen(self):
        try:
            self.ws = create_connection(self.connect_url)
            self.ws.send(json.dumps({
                "eventName": "join",
                "data": {
                    "playerNumber": self.player_number,
                    "playerName": self.player_name,
                    "token": self.token
                }
            }))

            while 1:
                result = self.ws.recv()
                msg = json.loads(result)
                event_name = msg["eventName"]
                data = msg["data"]
                self.takeAction(event_name, data)

        except Exception, e:
            print e.message
            self.doListen()


def main():
    argv_count = len(sys.argv)

    if argv_count > 2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]
        token = sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "Eric"
        player_number = 4
        token = "12345678"
        connect_url = "ws://localhost:8080/"

    sample_bot = LowPlayBot(player_name)
    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, sample_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
