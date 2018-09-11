#coding=UTF-8
from abc import abstractmethod

from pokerbot import PokerSocket
from pokerbot import LowPlayBot
from pokerbot import GreedyBot
from pokerbot import MonteCarloBot
from pokerbot import PomDPBot

import logging
import sys


class Log(object):
    def __init__(self,is_debug=True):
        self.is_debug=is_debug
        self.msg=None
        self.logger = logging.getLogger('hearts_logs')
        hdlr = logging.FileHandler('/log/hearts_logs.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        if is_debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def show_message(self,msg):
        if self.is_debug:
            print(msg)

    def save_logs(self,msg):
        self.logger.info(msg)


IS_DEBUG = False
system_log = Log(IS_DEBUG)


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
        connect_url="ws://localhost:3333/"
    
    #sample_bot=LowPlayBot(player_name, system_log)
    #sample_bot=GreedyBot(player_name, system_log)
    #sample_bot=MonteCarloBot(player_name, system_log)
    sample_bot = PomDPBot(player_name, system_log)

    myPokerSocket=PokerSocket(player_name,player_number,token,connect_url,sample_bot, system_log)
    myPokerSocket.doListen()


if __name__ == "__main__":
    main()
