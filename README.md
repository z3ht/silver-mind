# Silver Mind  
**Reinforcement learning chess AI**  
Designed By: Andrew Darling

![404](https://i.imgur.com/f0RThee.jpg)  
--- *GangstaRaptor by: arbsbuhpuh (https://imgur.com/gallery/d6vxE1f)* ---

<br>

**Goal:**  
The goal for creating Silver Mind is to build a Chess AI using reinforcement learning that can beat me 
(~1025 elo) at chess. A stretch goal is to beat Magnus Carlson at age 18 from the Magnus Carlson app having 
only trained on games from Garry Kasparov. This would be a great accomplishment because Magnus Carlson beat Kasparov 
at age 16 in 2004 and has only improved since then so Silver Mind would be required to learn a significant amount 
beyond its training.

**Strategy:**  
something something deep reinforcement learning something something DQN/PPO policy something something

<br>
---

#### Environment Setup:

Run `pip install .` to install all of the required libraries

Silver Mind must be trained with PGN files. PGN stands for Portable Game Notation and is the standard format for saving 
chess games. They can be downloaded from a variety of places off the internet. Save one file per game to a folder in 
the project directory called `pgns` then run `examples/train_silvermind.ipynb` to train Silver Mind.