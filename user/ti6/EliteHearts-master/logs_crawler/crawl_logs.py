#!/usr/bin/env python3
import argparse
import getpass
import json
import os
import sys
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

script_dir = os.path.dirname(os.path.realpath(__file__))
logs_dir = os.path.join(script_dir, 'logs')

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Fetch ranked game logs from official site')
  argparser.add_argument('min_id', metavar='N', type=int, nargs=1,
    help='Minimum ID of game logs (to prevent duplicated fetch)')
  args = argparser.parse_args()

  username = input('Username: trend\\').strip()
  password = getpass.getpass()
  min_game_id = args.min_id[0]

  session = requests.Session()

  resp = session.get('https://aicontest2018.trendmicro.com/login')
  if not resp.url.startswith('https://adfssts.trendmicro.com'):
    raise Exception(f'Expected ADFS login page, got "{resp.url}"')
  print('Redirecting to ADFS login page...')

  # login with ADFS
  adfs_login_url = resp.url
  payload = {
    'UserName': f'trend\\{username}',
    'Password': password,
    'AuthMethod': 'FormsAuthentication',
  }
  resp = session.post(adfs_login_url, data=payload)
  print('Logged into ADFS...')

  # callback to AI site
  html = BeautifulSoup(resp.text, 'html.parser')
  callback_url = html.form['action']
  payload = {
    html.form.input['name']: html.form.input['value'],
  }
  resp = session.post(callback_url, data=payload)
  if resp.status_code != 200:
    raise Exception(f'Expected 200 OK, got {resp.status_code} on "{resp.url}"')
  print('Callback to target site...')

  # enumerate game history
  records_url = 'https://aicontest2018.trendmicro.com/api/trend-hearts/ranked-games/?offset=0&limit=100&sortOrder=1&sortField=createTime'
  while records_url is not None:
    print(f'records_url="{records_url}"')
    resp = session.get(records_url)
    data = resp.json()
    for result in data['results']:
      if result['status'] != 2: continue
      result_id = result['id']
      result_url = f'https://aicontest2018.trendmicro.com/api/trend-hearts/ranked-games/{result_id}/'
      print(f'result_url="{result_url}"')
      resp2 = session.get(result_url)
      data2 = resp2.json()
      for game in data2['results'][0]['games']:
        if game['status'] != 2: continue
        game_id = game['id']
        # the games are enumerated in desc order -- if we pass the min_id, just exit!
        if game_id < min_game_id: sys.exit(0)
        game_url = f'https://aicontest2018.trendmicro.com/api/trend-hearts/game-logs/{game_id}/export/'
        print(f'game_url="{game_url}"')
        payload = {
          'json': json.dumps({'filename': f'{game_id}.json'}),
        }
        resp3 = session.post(game_url, data=payload)
        log_fpath = os.path.join(logs_dir, f'{game_id}.json')
        with open(log_fpath, 'w') as outfile:
          outfile.write(resp3.text)
    records_url = data['next']
