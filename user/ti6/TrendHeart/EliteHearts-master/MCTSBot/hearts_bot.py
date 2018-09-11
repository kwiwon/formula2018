#coding=UTF-8
from abc import abstractmethod

from websocket import create_connection
import json
import logging
import sys

# my library
from ai_agent import MCTSPlayer

class Log(object):
    def __init__(self,is_debug=True):
        self.is_debug=is_debug
        self.msg=None
        self.logger = logging.getLogger('hearts_logs')
        hdlr = logging.FileHandler('hearts_logs.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)
    def show_message(self,msg):
        if self.is_debug:
            print (msg)
    def save_logs(self,msg):
        self.logger.info(msg)

IS_DEBUG=True
system_log=Log(IS_DEBUG)


class PokerSocket(object):
    ws = ""
    def __init__(self,player_name,player_number,token,connect_url,poker_bot):
        self.player_name=player_name
        self.connect_url=connect_url
        self.player_number=player_number
        self.poker_bot=poker_bot
        self.token=token

    def takeAction(self, action, data):
       if action=="new_peer":
           self.poker_bot.set_all_player_name(data)
       elif action=="new_deal":
           self.poker_bot.process_new_deal(data)
       elif action=="pass_cards":
           pass_cards=self.poker_bot.choose_passing_cards(data)
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
           self.poker_bot.process_expose_cards(data)
       elif action=="new_round":
           self.poker_bot.process_new_round(data)
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
           system_log.show_message(message)
           system_log.save_logs(message)
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
           self.poker_bot.process_turn_end(data)
       elif action=="round_end":
           self.poker_bot.process_round_end(data)
       elif action=="game_end":
           self.poker_bot.game_over(data)
           self.ws.close()
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
                system_log.show_message(event_name)
                system_log.save_logs(event_name)
                #system_log.show_message(data)
                system_log.save_logs(data)
                self.takeAction(event_name, data)
        except Exception as e:
            system_log.show_message(e)
            system_log.save_logs(e)
            raise
            #self.doListen()

def main():
    argv_count=len(sys.argv)
    if argv_count>2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]
        token= sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name="Sample Bot"
        player_number=99
        token="12345678"
        connect_url="ws://localhost:8080/"
    #sample_bot=LowPlayBot(player_name)
    my_bot = MCTSPlayer(player_name)
    myPokerSocket=PokerSocket(player_name,player_number,token,connect_url,my_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
