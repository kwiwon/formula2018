#!/usr/bin/env pypy
# encoding: utf-8
import datetime
import glob
import json
import os
import re
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

'''
sys.path.insert(0, os.path.realpath(os.path.join(script_dir, '..', 'dummy-client')))
from deuces import Card, Evaluator

e = Evaluator()
print(e.evaluate_two(Card.gen_cards(['3H', '5S'])))
print(e.evaluate_three(Card.gen_cards(['3H', '5S']), Card.gen_cards(['5H', '6H', '7H'])))
print(e.evaluate_four(Card.gen_cards(['3H', '5S']), Card.gen_cards(['5H', '6H', '7H', 'AC'])))
print(e.evaluate_five(Card.gen_cards(['3H', '5S']), Card.gen_cards(['5H', '6H', '7H', 'AC', '2D'])))
'''

# ------------------------------------------------------------------------------


# in: [2018-07-19T11:58:58.657] [INFO] userProdu...
# out: timestamp
TIMESTAMP_LOG_LINE_REGEX = re.compile(r'^\[([\d\-T:\.]+)+\]')

# in: [2018-07-19T11:58:58.657] [INFO] userProductionLog - 2018-07-19 11:58:58 657: >>> event ROUND_END >>> {"event...
# in: [2018-08-25T13:53:41.918] [INFO] guestProductionLog - {"eventName":"new_peer","data":{".....
# out: timestamp, event_name, event
# EVENT_LOG_LINE_REGEX = re.compile(r'\[([\d\-T:\.]+)+\] .* >>> event ([\w_]+) >>> (.+)')
EVENT_LOG_LINE_REGEX = re.compile(r'\[([\d\-T:\.]+)+\] .* - (.+)')

# in: [2018-07-24T15:56:39.345] [INFO] userProductionLog - 2018-07-24 15:56:39 345: >>> table 267 >>> new round : 26
# out: timestamp, table_num, round_num
TABLE_LOG_LINE_REGEX = re.compile(r'\[([\d\-T:\.]+)+\] .* >>> table (\d+) >>> new round : (\d+)')

# ------------------------------------------------------------------------------

dealLog = {}


class Deal(object):
    def __init__(self, deal_num):
        self.deal_num = deal_num
        self.player = {}

    def get_player(self, player_num):
        return self.player.get(player_num, None)

    def new_player(self, player_num):
        self.player[player_num] = Person(player_num)

    def to_dict(self):
        player = sorted(self.player.items(), key=lambda t: t[0])
        return {
            'dealNumber': self.deal_num,
            'player': map(lambda t: t[1].to_dict(), player),
        }


class Person(object):
    def __init__(self, player_num):
        self.player_num = player_num
        self.receivedCard = None
        self.rounds = {}

    def get_round(self, round_num):
        return self.rounds.get(round_num, None)

    def new_round(self, round_num):
        self.rounds[round_num] = Round(round_num)

    def add_receive_cards(self, event):
        self.receivedCard = event

    def to_dict(self):
        rounds = sorted(self.rounds.items(), key=lambda t: t[0])
        return {
            'playerNumber': self.player_num,
            'receivedCard': self.receivedCard,
            'rounds': map(lambda t: t[1].to_dict(), rounds),
        }


class Round(object):
    def __init__(self, round_num):
        self.round_num = round_num

        self.roundCard = None
        self.score = 0

    def add_round_card(self, card):
        self.roundCard = card

    def add_score(self, score):
        self.score = score

    def to_dict(self):
        return {
            'roundNumber': self.round_num,
            'roundCard': self.roundCard,
            'score': self.score,
        }


# ------------------------------------------------------------------------------


def timestamp_to_datetime(timestamp):
    return datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')


def mystrftime(dt):
    return dt.strftime('%Y%m%d-%H%M%S')


def parse_timestamp_from_log_line(line):
    timestamp = TIMESTAMP_LOG_LINE_REGEX.findall(line)[0]
    return timestamp_to_datetime(timestamp)


def parse_new_round_from_log_line(line):
    time, table_num, round_num = TABLE_LOG_LINE_REGEX.findall(line)[0]
    return (time, int(table_num), int(round_num))


def parse_event_from_log_line(line):
    time, event = EVENT_LOG_LINE_REGEX.findall(line)[0]
    return (time, json.loads(event))


def handle_event(time, event):
    # ignore ghost table-rounds ...
    eventName = event['eventName']
    print eventName
    if eventName == 'new_deal':
        handle_new_deal(time, event)
    elif eventName == 'round_end':  # 13 turn card
        handleRound(event)
    elif eventName == 'expose_cards_end':
        handlePassCards(event)
    elif eventName == 'game_end':
        handleGameEnd(event)


def handleRound(event):
    dealNum = int(event['data']['dealNumber'])
    roundNum = int(event['data']['roundNumber'])
    deal = dealLog.get(dealNum, None)
    if not deal: return

    for player in event['data']['players']:
        playerNum = player['playerNumber']

        person = deal.get_player(playerNum)
        if not person: return

        person.new_round(roundNum)
        round = person.get_round(roundNum)
        if not round: return

        round.add_round_card(player['roundCard'])
        round.add_score(player['dealScore'])


def handlePassCards(event):
    dealNum = int(event['data']['dealNumber'])
    for player in event['data']['players']:
        playerNum = player['playerNumber']
        dealLog[dealNum] = Deal(dealNum)
        dealLog[dealNum].new_player(playerNum)

        person = dealLog[dealNum].get_player(playerNum)
        if not person: return

        person.add_receive_cards(player['receivedCards'])


def handle_new_deal(time, event):
    print "newdeal"


def handleGameEnd(event):
    for deal in dealLog:
        table_fname = 'Game.json'
        table_fpath = os.path.join(script_dir, table_fname)
        with open(table_fpath, 'a') as outfile:
            outfile.write(json.dumps(dealLog[deal].to_dict(), indent=2))
        # print(table_fname)


def parse_log_lines(lines):
    for line in lines:
        time, event = parse_event_from_log_line(line)
        handle_event(time, event)


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    data_dir = os.path.realpath(os.path.join(script_dir, '..', 'data'))
    logs_path = os.path.join(data_dir, 'guest.log')
    print('logs_dir: {}'.format(logs_path))

    lines = []
    with open(logs_path) as infile:
        lines += [line for line in infile if line]
    lines = sorted(lines, key=parse_timestamp_from_log_line)

    parse_log_lines(lines)
