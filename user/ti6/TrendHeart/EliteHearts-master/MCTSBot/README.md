# Monte-Carlo-Tree-Search Bot

```
Spirit of MCTS:   
Traverse along promising nodes →→ Expand new node →→ Simulation playout →→ Back propagation
                           ↑                                                 ↓
                           ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

Based on https://github.com/Devking/HeartsAI/blob/master/MCTSPlayer.java, we revise and enhance with some preferred feature and strategy.

## Version Histrory
- 2018/9/4 - First commit of v1.0


## Version Strategy
v1.0
- Remember possible hand cards of other players
- Randomly choose three cards when exchanging cards
- Randomly expose AH
- Randomly choose from candidate cards for all players in simulation playout
- Choose best UCT score of only myself when traversing game tree
- Simulation efficiency is about 0.001~0.003s from very beginning of one deal to the end

## Execute
1. Open local server, `trend-hearts-1.0.0-alpha.7-macos`
2. `python hearts_bot.py <player_name> <player_number> <token> ws://localhost:8080/`


