import json
import sys

from websocket import create_connection

class PokerSocket(object):
    ws = ""
    def __init__(self,player_name,player_number,token,connect_url,poker_bot,system_log):
        self.player_name=player_name
        self.connect_url=connect_url
        self.player_number=player_number
        self.poker_bot=poker_bot
        self.token=token
        self.system_log=system_log

    def takeAction(self,action, data):
       if  action=="new_deal":
           self.poker_bot.new_deal(data)
       elif action=="pass_cards":
           pass_cards=self.poker_bot.pass_cards(data)
           self.ws.send(json.dumps(
                {
                    "eventName": "pass_my_cards",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "cards": pass_cards
                    }
                }))
       elif action=="receive_opponent_cards":
           self.poker_bot.receive_opponent_cards(data)
       elif action=="expose_cards":
           export_cards = self.poker_bot.expose_my_cards(data)
           if export_cards!=None:
               self.ws.send(json.dumps(
                   {
                       "eventName": "expose_my_cards",
                       "data": {
                           "dealNumber": data['dealNumber'],
                           "cards": export_cards
                       }
                   }))
       elif action=="expose_cards_end":
           self.poker_bot.expose_cards_end(data)
       elif action=="your_turn":
           pick_card = self.poker_bot.pick_card(data)
           message="Send message:{}".format(json.dumps(
                {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))
           self.system_log.show_message(message)
           self.system_log.save_logs(message)
           self.ws.send(json.dumps(
               {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))
       elif action=="turn_end":
           self.poker_bot.turn_end(data)
       elif action=="round_end":
           self.poker_bot.round_end(data)
       elif action=="deal_end":
           self.poker_bot.deal_end(data)
           self.poker_bot.reset_card_his()
       elif action=="game_end":
           self.poker_bot.game_over(data)
           self.ws.close()
       elif action == "new_round":
           self.poker_bot.new_round(data)

    def doListen(self):
        try:
            self.ws = create_connection(self.connect_url)
            self.ws.send(json.dumps({
                "eventName": "join",
                "data": {
                    "playerNumber":self.player_number,
                    "playerName":self.player_name,
                    "token":self.token
                }
            }))
            while 1:
                result = self.ws.recv()
                msg = json.loads(result)
                event_name = msg["eventName"]
                data = msg["data"]
                self.system_log.show_message(event_name)
                self.system_log.save_logs(event_name)
                self.system_log.show_message(data)
                self.system_log.save_logs(data)
                self.takeAction(event_name, data)
        except Exception as e:
            self.system_log.show_message(e)
            self.system_log.save_logs(e)
            self.system_log.save_logs(sys.exc_info())
            self.doListen()
            #raise e
