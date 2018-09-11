# Trend Hearts

[![works badge](https://cdn.rawgit.com/nikku/works-on-my-machine/v0.2.0/badge.svg)](https://github.com/nikku/works-on-my-machine)

**Table of Contents**
- [Trend Hearts](#trend-hearts)
  * [Compatibility](#compatibility)
  * [Simple Usage](#simple-usage)
    + [For game server](#for-game-server)
    + [For local emulator](#for-local-emulator)
  * [Render](#render)
  * [Emulator Message](#emulator-message)
    + [new_game](#new-game)
    + [new_deal](#new-deal)
    + [pass_cards](#pass-cards)
    + [receive_opponent_cards](#receive-opponent-cards)
    + [pass_cards_end](#pass-cards-end)
    + [expose_cards](#expose-cards)
    + [expose_cards_end](#expose-cards-end)
    + [new_round](#new-round)
    + [turn_end](#turn-end)
    + [your_turn](#your-turn)
    + [round_end](#round-end)
    + [deal_end](#deal-end)
    + [game_end](#game-end)


## Compatibility
This environment is designed to emulate the game server. It uses same game rules as the game server and will send out same message as the game server does.

## Simple Usage
### For game server

Call `connector.listen()`
```python
env = TrendHeartsEnv()
env.reset()
connect_url = 'ws://127.0.0.1:8888/'

# method 1: to join online server
player_name = f'random_agent'
player_number = 1
token = 12345678
agent = RandomAgent(player_name=player_name)
connector = Connector(player_name, player_number, token, connect_url, agent)
connector.listen()
```

### For local emulator
Pass `connector` to `env` and call `env.reset()` and `env.game_start()`

```python
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
    message['data']['playerNumber'] = player_number
    message['data']['playerName'] = player_name
    message['data']['token'] = token
    message = json.dumps(message)
    env.join(connector, message)


game_num = 1
for i in range(game_num):
    env.reset()
    env.game_start()
```

## Render
Explicitly pass `render=True` to show each round result
```python
env = TrendHeartsEnv(render=True)
```

Example output:
```
deal1 round1
player4 [K♦][T♦][Q♣][A♥][9♦][K♥][3♠][3♣][J♦][4♦][6♠][5♣]    deal score:0 game score:0
player1 [9♠][T♠][7♦][Q♦][7♠][8♦][5♦][2♦][J♠][J♣][8♠][5♥]    deal score:0 game score:0
player2 [Q♠][K♠][K♣][A♠][8♥][3♥][7♣][6♥][4♠][4♥][5♠][8♣]    deal score:0 game score:0
player3 [2♥][Q♥][3♦][T♥][9♣][9♥][6♦][7♥][A♦][J♥][4♣][2♠] *  deal score:0 game score:0
[2♣][6♣][T♣][A♣]
```
From top to bottom are the hand cards of north, east, south, west player.
The last line shows the round cards played by north ,east, south, and west player.
Line with an asterisk indicates the round winner.

## Emulator Message
Emulator sends same events to agent as the game server does. For debugging purpose, each agent will be able to see others private information, such as hand cards.

### new_game
```json
{
  "eventName": "new_game",
  "data": {
    "players": [
        {
            "playerNumber": 1,
            "playerName": "player1",
            "status": 0
        },
        {
            "playerNumber": 2,
            "playerName": "player2",
            "status": 0
        },
        {
            "playerNumber": 3,
            "playerName": "player3",
            "status": 0
        },
        {
            "playerNumber": 4,
            "playerName": "player4",
            "status": 0
        }
    ]
  }
}
```

### new_deal
```json
{
  "eventName": "new_deal",
  "data": {
    "dealNumber": 3,
    "self": {
      "playerNumber": 4,
      "playerName": "player4",
      "gameScore": -4,
      "dealScore": 0,
      "cards": [
        "5C",
        "QC",
        "2S",
        "2C",
        "TH",
        "6C",
        "6H",
        "9S",
        "4H",
        "8H",
        "3D",
        "2D",
        "JC"
      ],
      "cardsCount": 13
    },
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "cards": [
          "5C",
          "QC",
          "2S",
          "2C",
          "TH",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3D",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "4S",
          "9C",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "JS",
          "5S"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "cards": [
          "TC",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "KC",
          "7D",
          "3S",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "KD",
          "KH",
          "7S",
          "7H",
          "5D",
          "8S",
          "TS"
        ],
        "cardsCount": 13
      }
    ]
  }
}
```

### pass_cards
```json
{
  "eventName": "pass_cards",
  "data": {
    "dealNumber": 3,
    "self": {
      "playerNumber": 4,
      "playerName": "player4",
      "gameScore": -4,
      "dealScore": 0,
      "cards": [
        "5C",
        "QC",
        "2S",
        "2C",
        "TH",
        "6C",
        "6H",
        "9S",
        "4H",
        "8H",
        "3D",
        "2D",
        "JC"
      ],
      "cardsCount": 13
    },
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "cards": [
          "5C",
          "QC",
          "2S",
          "2C",
          "TH",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3D",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "4S",
          "9C",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "JS",
          "5S"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "cards": [
          "TC",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "KC",
          "7D",
          "3S",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "KD",
          "KH",
          "7S",
          "7H",
          "5D",
          "8S",
          "TS"
        ],
        "cardsCount": 13
      }
    ],
    "receiver": "player2"
  }
}
```

### receive_opponent_cards
```json
{
  "eventName": "receive_opponent_cards",
  "data": {
    "dealNumber": 3,
    "self": {
      "playerNumber": 4,
      "playerName": "player4",
      "gameScore": -4,
      "dealScore": 0,
      "receivedCards": [
        "KC",
        "TC",
        "3S"
      ],
      "receivedFrom": "player2",
      "cards": [
        "5C",
        "QC",
        "2S",
        "2C",
        "TH",
        "6C",
        "6H",
        "9S",
        "4H",
        "8H",
        "3D",
        "2D",
        "JC"
      ],
      "cardsCount": 13
    },
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "receivedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedFrom": "player2",
        "cards": [
          "5C",
          "QC",
          "2S",
          "2C",
          "TH",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3D",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "receivedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedFrom": "player3",
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "4S",
          "9C",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "JS",
          "5S"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "receivedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedFrom": "player4",
        "cards": [
          "TC",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "KC",
          "7D",
          "3S",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "receivedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedFrom": "player1",
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "KD",
          "KH",
          "7S",
          "7H",
          "5D",
          "8S",
          "TS"
        ],
        "cardsCount": 13
      }
    ]
  }
}
```

### pass_cards_end
```json
{
  "eventName": "pass_cards_end",
  "data": {
    "dealNumber": 3,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "pickedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedFrom": "player2",
        "cards": [
          "5C",
          "QC",
          "2S",
          "2C",
          "TH",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3D",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "pickedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedFrom": "player3",
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "4S",
          "9C",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "JS",
          "5S"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "pickedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedFrom": "player4",
        "cards": [
          "TC",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "KC",
          "7D",
          "3S",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "pickedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedFrom": "player1",
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "KD",
          "KH",
          "7S",
          "7H",
          "5D",
          "8S",
          "TS"
        ],
        "cardsCount": 13
      }
    ]
  }
}
```

### expose_cards
```json
{
  "eventName": "expose_cards",
  "data": {
    "dealNumber": 3,
    "self": {
      "playerNumber": 1,
      "playerName": "player1",
      "gameScore": -6,
      "dealScore": 0,
      "cards": [
        "JH",
        "QH",
        "6S",
        "9D",
        "AH",
        "7C",
        "3C",
        "3H",
        "JD",
        "7H",
        "5D",
        "8S",
        "JS"
      ],
      "cardsCount": 13,
      "candidateCards": [
        "AH"
      ]
    },
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "KC",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "cards": [
          "TH",
          "2C",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "9C",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 13
      }
    ]
  }
}
```

### expose_cards_end
```json
{
  "eventName": "expose_cards_end",
  "data": {
    "dealNumber": 3,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "pickedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedFrom": "player2",
        "exposedCards": [],
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "KC",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "pickedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedFrom": "player3",
        "exposedCards": [],
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "pickedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedFrom": "player4",
        "exposedCards": [],
        "cards": [
          "TH",
          "2C",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "pickedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedFrom": "player1",
        "exposedCards": [],
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "9C",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 13
      }
    ]
  }
}
```

### new_round
```json
{
  "eventName": "new_round",
  "data": {
    "dealNumber": 3,
    "roundNumber": 1,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "KC",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "TH",
          "2C",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "9C",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 13
      }
    ],
    "roundPlayers": [
      "player2",
      "player3",
      "player4",
      "player1"
    ]
  }
}
```

### turn_end
```json
{
  "eventName": "turn_end",
  "data": {
    "dealNumber": 3,
    "roundNumber": 1,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "KC",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 13,
        "serverRandom": true
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 13,
        "serverRandom": true
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "TH",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 12,
        "serverRandom": true,
        "roundCard": "2C"
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "9C",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 13,
        "serverRandom": true
      }
    ],
    "roundPlayers": [
      "player2",
      "player3",
      "player4",
      "player1"
    ],
    "turnPlayer": "player2",
    "turnCard": "2C"
  },
  "serverRandom": true
}
```

### your_turn
```json
{
  "eventName": "your_turn",
  "data": {
    "dealNumber": 3,
    "roundNumber": 1,
    "self": {
      "playerNumber": 2,
      "playerName": "player2",
      "gameScore": -48,
      "dealScore": 0,
      "exposedCards": [],
      "cards": [
        "TH",
        "2C",
        "TD",
        "AS",
        "AC",
        "4C",
        "9H",
        "KS",
        "3D",
        "7D",
        "8D",
        "QD",
        "8C"
      ],
      "cardsCount": 13,
      "candidateCards": [
        "2C"
      ]
    },
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "KC",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 13,
        "serverRandom": true
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 13,
        "serverRandom": true
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "TH",
          "2C",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 13,
        "serverRandom": true
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "9C",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 13,
        "serverRandom": true
      }
    ],
    "roundPlayers": [
      "player2",
      "player3",
      "player4",
      "player1"
    ]
  }
}
```

### round_end
```json
{
  "eventName": "round_end",
  "data": {
    "dealNumber": 3,
    "roundNumber": 1,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "5C",
          "QC",
          "2S",
          "TC",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3S",
          "2D",
          "JC"
        ],
        "cardsCount": 12,
        "serverRandom": true,
        "roundCard": "KC"
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "AH",
          "3C",
          "3H",
          "JD",
          "7H",
          "5D",
          "8S",
          "JS"
        ],
        "cardsCount": 12,
        "serverRandom": true,
        "roundCard": "7C"
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "TH",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "3D",
          "7D",
          "8D",
          "QD",
          "8C"
        ],
        "cardsCount": 12,
        "serverRandom": true,
        "roundCard": "2C"
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "dealScore": 0,
        "scoreCards": [],
        "exposedCards": [],
        "cards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "4S",
          "KD",
          "5S",
          "KH",
          "7S",
          "TS"
        ],
        "cardsCount": 12,
        "serverRandom": true,
        "roundCard": "9C"
      }
    ],
    "roundPlayers": [
      "player2",
      "player3",
      "player4",
      "player1"
    ]
  },
  "roundPlayer": "player4"
}
```

### deal_end
```json
{
  "eventName": "deal_end",
  "data": {
    "dealNumber": 3,
    "roundNumber": 13,
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -4,
        "errorCount": 39,
        "timeoutCount": 0,
        "dealScore": 0,
        "scoreCards": [
          "TC"
        ],
        "pickedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedFrom": "player2",
        "exposedCards": [],
        "shootingTheMoon": false,
        "status": 0,
        "initialCards": [
          "5C",
          "QC",
          "2S",
          "2C",
          "TH",
          "6C",
          "6H",
          "9S",
          "4H",
          "8H",
          "3D",
          "2D",
          "JC"
        ]
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "errorCount": 39,
        "timeoutCount": 0,
        "dealScore": 0,
        "scoreCards": [],
        "pickedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedFrom": "player3",
        "exposedCards": [],
        "shootingTheMoon": false,
        "status": 0,
        "initialCards": [
          "JH",
          "QH",
          "6S",
          "9D",
          "4S",
          "9C",
          "AH",
          "7C",
          "3C",
          "3H",
          "JD",
          "JS",
          "5S"
        ]
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -48,
        "errorCount": 39,
        "timeoutCount": 0,
        "dealScore": -7,
        "scoreCards": [
          "JH",
          "5H",
          "2H",
          "6H",
          "4H",
          "9H",
          "3H"
        ],
        "pickedCards": [
          "KC",
          "TC",
          "3S"
        ],
        "receivedCards": [
          "3D",
          "2C",
          "TH"
        ],
        "receivedFrom": "player4",
        "exposedCards": [],
        "shootingTheMoon": false,
        "status": 0,
        "initialCards": [
          "TC",
          "TD",
          "AS",
          "AC",
          "4C",
          "9H",
          "KS",
          "KC",
          "7D",
          "3S",
          "8D",
          "QD",
          "8C"
        ]
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -18,
        "errorCount": 39,
        "timeoutCount": 0,
        "dealScore": -19,
        "scoreCards": [
          "QH",
          "TH",
          "AH",
          "KH",
          "7H",
          "QS",
          "8H"
        ],
        "pickedCards": [
          "7H",
          "5D",
          "8S"
        ],
        "receivedCards": [
          "4S",
          "9C",
          "5S"
        ],
        "receivedFrom": "player1",
        "exposedCards": [],
        "shootingTheMoon": false,
        "status": 0,
        "initialCards": [
          "6D",
          "5H",
          "QS",
          "2H",
          "4D",
          "AD",
          "KD",
          "KH",
          "7S",
          "7H",
          "5D",
          "8S",
          "TS"
        ]
      }
    ]
  }
}
```

### game_end
```json
{
  "eventName": "game_end",
  "data": {
    "players": [
      {
        "playerNumber": 4,
        "playerName": "player4",
        "gameScore": -23,
        "errorCount": 52,
        "timeoutCount": 0,
        "rank": 2,
        "deals": [
          {
            "dealNumber": 1,
            "score": -4,
            "exposedCards": []
          },
          {
            "dealNumber": 2,
            "score": 0,
            "exposedCards": []
          },
          {
            "dealNumber": 3,
            "score": 0,
            "exposedCards": []
          },
          {
            "dealNumber": 4,
            "score": -19,
            "exposedCards": []
          }
        ]
      },
      {
        "playerNumber": 1,
        "playerName": "player1",
        "gameScore": -6,
        "errorCount": 52,
        "timeoutCount": 0,
        "rank": 1,
        "deals": [
          {
            "dealNumber": 1,
            "score": 0,
            "exposedCards": []
          },
          {
            "dealNumber": 2,
            "score": -6,
            "exposedCards": []
          },
          {
            "dealNumber": 3,
            "score": 0,
            "exposedCards": []
          },
          {
            "dealNumber": 4,
            "score": 0,
            "exposedCards": []
          }
        ]
      },
      {
        "playerNumber": 2,
        "playerName": "player2",
        "gameScore": -62,
        "errorCount": 52,
        "timeoutCount": 0,
        "rank": 4,
        "deals": [
          {
            "dealNumber": 1,
            "score": -14,
            "exposedCards": []
          },
          {
            "dealNumber": 2,
            "score": -34,
            "exposedCards": []
          },
          {
            "dealNumber": 3,
            "score": -7,
            "exposedCards": []
          },
          {
            "dealNumber": 4,
            "score": -7,
            "exposedCards": []
          }
        ]
      },
      {
        "playerNumber": 3,
        "playerName": "player3",
        "gameScore": -37,
        "errorCount": 52,
        "timeoutCount": 0,
        "rank": 3,
        "deals": [
          {
            "dealNumber": 1,
            "score": -15,
            "exposedCards": []
          },
          {
            "dealNumber": 2,
            "score": -3,
            "exposedCards": []
          },
          {
            "dealNumber": 3,
            "score": -19,
            "exposedCards": []
          },
          {
            "dealNumber": 4,
            "score": 0,
            "exposedCards": []
          }
        ]
      }
    ]
  }
}
```