#!/usr/bin/env pypy
# encoding: utf-8
import datetime
import json
import os
import re
import csv
import sys
import argparse
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))

### TODO - Make user can easily extract features at decision points:
# Add game_end score, score_cards_list, lackSuits, playedCards

common_header = ['gameNumber', 'dealNumber', 'roundNumber', 'turnNumber', 'isotime', 'eventName']
game_info_header = [
    'playerNumber', 'playerName', 'gameScore', 'dealScore',
    'cards', 'cardsCount', 'scoreCards', 'pickedCards', 'receivedCards',
    'receivedFrom', 'exposedCards', 'initialCards', 'roundCard',
    'serverRandom', 'timeoutCount', 'errorCount', 'status',
    'shootingTheMoon', 'candidateCards'
]
position = ['N', 'E', 'S', 'W']

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


def timestamp_to_datetime(timestamp):
    return datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')


def mystrftime(dt):
    return dt.strftime('%Y%m%d-%H%M%S')


def parse_timestamp_from_log_line(line):
    timestamp = TIMESTAMP_LOG_LINE_REGEX.findall(line)[0]
    return timestamp_to_datetime(timestamp)


def parse_new_round_from_log_line(line):
    time, table_num, round_num = TABLE_LOG_LINE_REGEX.findall(line)[0]
    return time, int(table_num), int(round_num)


def parse_event_from_log_line(line):
    time, event = EVENT_LOG_LINE_REGEX.findall(line)[0]
    return time, json.loads(event)


def handle_event(event, time=''):
    # ignore ghost table-rounds ...
    event_name = event['eventName']
    parsed_event = []
    if event_name in ['new_deal', 'pass_cards_end', 'expose_cards_end', 'turn_end', 'round_end', 'deal_end' ]:
        parsed_event = event_parser(event, event_name, time=time)
    elif event_name == 'game_end':
        pass
    #    parsed_event= handleGameEnd(event, event_name)
    return parsed_event


def event_parser(event, event_name, time=''):
    data = event['data']
    players = []
    player_turn_map = {}

    if event_name in ['turn_end', 'round_end']:
        round_players = data['roundPlayers']
        for turn_num in range(4):
            player_turn_map[round_players[turn_num]] = turn_num + 1

    game_num, deal_num, round_num = data['gameNumber'], data['dealNumber'], data['roundNumber'],
    for player in data['players']:
        turn_number = None
        if len(player_turn_map) > 0:
            turn_number = player_turn_map[player['playerName']]

        player_info = []
        for key in game_info_header:
            player_info.append(player.get(key, None))
        players.append([game_num, deal_num, round_num, turn_number, time, event_name] + player_info)
    return players


def parse_log(log_path, output_path, log_type):
    if log_type == 'guest':
        parse_guest_log(log_path, output_path)
    elif log_type == 'server':
        parse_server_log(log_path, output_path)

    post_processing(output_path)


def parse_guest_log(log_path, output_path):
    lines = []
    with open(log_path) as infile:
        lines += [line for line in infile if line]

    with open(output_path, 'w') as csvfile:
        log_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(common_header + game_info_header)
        for line in lines:
            time, event = parse_event_from_log_line(line)
            parsed_line = handle_event(event, time)

            for p in parsed_line:
                log_writer.writerow(p)


def parse_server_log(log_path, output_path):
    with open(log_path, 'rb') as f:
        server_log = json.load(f)
        events = server_log['events']

    with open(output_path, 'w') as csvfile:
        log_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(common_header + game_info_header)
        for event in events:
            parsed_event = handle_event(event)

            for p in parsed_event:
                log_writer.writerow(p)


def post_processing(output_path):
    df = pd.read_csv(output_path)
    df = add_position(df)
    df = add_deal_rank(df)
    df = add_initial_cards(df)
    df.to_csv(output_path+'.csv')


def add_position(df):
    for i in range(4):
        curr_game_num = i+1
        rotated_position = position[i:] + position[:i]
        df.loc[(df.gameNumber == curr_game_num), 'position'] = rotated_position * (len(df[df.gameNumber == curr_game_num]) // 4)
    return df


def add_deal_rank(df):
    for i in range(4):
        for j in range(4):
            curr_game_num = i+1
            df_round_end = df[(df.gameNumber == curr_game_num) & (df.dealNumber == j+1) & (df.roundNumber == 13) & (df.eventName == 'round_end')].copy()
            df_round_end.loc[:, 'rank'] = df_round_end.dealScore.rank(ascending=False, method='min')
            rank_list = df_round_end['rank'].tolist()
            rank_list = [int(r) for r in rank_list]
            df.loc[(df.gameNumber == curr_game_num), 'dealEndRank'] = rank_list * (len(df[df.gameNumber == curr_game_num]) // 4)
    return df


def add_initial_cards(df):
    for i in range(4):
        curr_game_num = i + 1
        for j in range(4):
            curr_deal_num = j + 1
            initial_cards = df[(df.eventName == 'deal_end') & (df.dealNumber == curr_deal_num)].initialCards.tolist()
            num_rows = len(df[(df.gameNumber == curr_game_num) & (df.dealNumber == curr_deal_num)])
            df.loc[(df.gameNumber == curr_game_num) & (df.dealNumber == curr_deal_num), 'initialCards'] = initial_cards * (num_rows // 4)
    return df


def main():
    """
    Usage:
        python hearts_log_parser.py [-h] --log_path LOG_PATH --output_path OUTPUT_PATH [--log_type {server,guest}]
        python hearts_log_parser.py [-h] -i LOG_PATH -o OUTPUT_PATH [-t {server,guest}]
    """
    parser = argparse.ArgumentParser(description='Hearts log parser')
    parser.add_argument('--log_path', '-i', required=True, help='path to SDK guest log')
    parser.add_argument('--output_path', '-o', required=True, help='path to output (format: csv)')
    parser.add_argument('--log_type', '-t', choices=['server', 'guest'], default='server', help='log_type')
    args = parser.parse_args()

    log_path = args.log_path
    output_path = args.output_path
    log_type = args.log_type

    parse_log(log_path, output_path, log_type)


if __name__ == '__main__':
    sys.exit(main())
