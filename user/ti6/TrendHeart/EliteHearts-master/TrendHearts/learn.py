from hearts import TrendHeartsEnv
from agent import RandomAgent
from agent.utils import Connector
import json


if __name__ == '__main__':

    env = TrendHeartsEnv(render=False)
    env.reset()
    connect_url = 'ws://127.0.0.1:8888/'  # for compatibility

    # method 1: to join online server
    # player_name = f'random_agent'
    # player_number = 1
    # token = 12345678
    # agent = RandomAgent(player_name=player_name)
    # connector = Connector(player_name, player_number, token, connect_url, agent)
    # connector.listen()

    # method 2: to use local simulator
    # create four dummies to join
    for i in range(4):
        player_name = f'random_agent_{i+1}'
        player_number = i + 1
        token = 12345678 + i
        agent = RandomAgent(player_name=player_name)
        connector = Connector(player_name, player_number, token, connect_url, agent, local=True)

        # emulate join event
        message = {'eventName': 'join', 'data': {}}
        message['data']['playerNumber'] = i + 1
        message['data']['playerName'] = f'player{i + 1}'
        message['data']['token'] = 12345678 + i
        message = json.dumps(message)
        env.join(connector, message)


    game_num = 1
    for i in range(game_num):
        env.reset()
        env.game_start()