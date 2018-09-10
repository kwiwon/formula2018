# POMCP Bot

This is a porting from the [html5-hearts](https://github.com/yyjhao/html5-hearts/blob/master/js/PomDPBrain.js) repo.

  * Assuming all other players to be playing using the greedy strategy, the game can then be formulated as a POMDP and can thus be solved with the POMCP Algorithm. This brain implements the POMCP algorithm.)
  * related paper [nips 2010](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)
  * [report of the author of the repo](https://yjyao.com/res/hearts.pdf)


## Current Feature

* greedy algorithm at `pokerbot/greedy`
  * a rule based algorithm, detail can be found at the user's [report](https://yjyao.com/res/hearts.pdf)

* monte-carlo algorithm at `pokerbot/montecarlo`
  * similar to [NYU's lookahead algorithm](https://github.com/Devking/HeartsAI/blob/master/LookAheadPlayer.java)
  * the different part is that the author assume other player take the greedy stratgy mentioned above
  * details can be found at the user's [report](https://yjyao.com/res/hearts.pdf)

* pomcp algorithm at `pokerbot/pomdp`
  * POMCP stands for [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)
  * the author assume the state is the current player’s turn (details in his [report](https://yjyao.com/res/hearts.pdf)).
  * based on his code, we think 
    * the observation is the list of (player draw the card, card) between each state (so the cards will across different rounds)
    * the state is a complete information comes from the sampled cards.
    * the tree constains action nodes and observation nodes, the root of the tree is a dummy action node
      * based on different observations, we decide we should move to which observation node
      * at each observation node, we decide an action to take based on past learned values(scores/rewards) and a weight to less visited actions.
      * then used a simulator to get the next observations and recursively go the children nodes based on the next observations (assume other players use greedy). after that we will get reward of the picked action and update the values and visit counts of the node.
      * if the observation is new, we use rollout strategy to play the game to the end and return the reward (assume other players use greedy)
    * each time in our turn, we can sample opponent's card many times and find the current observations node and start to estimate the values (scores/rewards) based on the simulation results.
    * after many times of simulation, we can just pick a best action from current observation node. 

  * different part from the author's code is that 
    * we will directly initial action nodes for the observation nodes which is the directly child of the root node 
      since at this level, the return reward score from rollout will not be update to any upper level nodes.
    * we keep the root node in a dictionary with keys generated from informations we collected in our first turn, so that in the future, you may save to tree and load the tree again to accumulate your game history.
      
* Exchange three cards stage: rule-based applied, just copied from `sample_bot.py`
* Expose AH stage: random applied, just copied from `sample_bot.py`


## Usage

### Requirements
* python 3.6
* websocket-client==0.52.0

### To test at local

1. revise player.py

2. open local game server, choose a player as empty seat

3. run your player

```
python player4debug.py <player_name> <player_number> <token> ws://<ip of game server>:<port>/
```

4. find logs at `hearts_log.log`


### to put on the game server

#### build image
```
make build_image
```

#### test with local server
```
make run IP=x.x.x.x
```
* $PO: port, default "3333"
* $IP: ip of the game server
* $P: player name, default "Sample Bot"
* $N: player number, default "99"
* $T: token, default "12345678"

after running the logs can found at `$(PWD)/log/hearts_log.log`

#### upload to official docker server
```
make build TEAM=xxx
```
xxx is your team number

the script will 
1. build docker image
2. tag with practice and rank
3. prompt you to login with your AD account
4. push the docker image to the game server

