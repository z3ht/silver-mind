# Silver Mind  
**Reinforcement learning chess AI**  
Designed By: Andrew Darling

![404](https://i.imgur.com/f0RThee.jpg)  
--- *GangstaRaptor by: arbsbuhpuh (https://imgur.com/gallery/d6vxE1f)* ---

<br>

**Goal:**  
The goal for creating Silver Mind is to build a Chess AI that can beat me (~1025 elo) at chess. A stretch goal is to beat Magnus Carlson at age 18 from the Magnus Carlson app having only trained on games from Garry Kasparov and other GMs. This would be a great accomplishment because Magnus Carlson beat Kasparov at age 16 in 2004 and has only improved since then so Silver Mind would be required to learn a significant amount beyond its training.

**Strategy:**  
Develop two algorithms: Silver Mind and Gold Mind. Silver Mind will be built from training data and should know possible moves it has available. Gold Mind should be capable of making any move (even illegal ones) but learn which are possible and more so learn which ones will be strong

Silver Mind:
1) Create deep learning model that predicts white win (1)/black win (-1) probability given a board state
2) Generate available moves
3) Move to where the future state will have the highest win probability
	Representation:
		- White: 1
		- Draw: 0
		- Black: -1
4) Continue until victory

Definitions:

Chess State:
A few options - `https://en.wikipedia.org/wiki/Board_representation_(computer_chess)`

Things to represent:
 - 8x8 Board: Data about each spot (8 possible states)
 - Turn

Things to ignore:
 - 50-move draw rule (I will not factor this into my model)
 - 3 move surrender (I will not factor this into my model)
 - Possible attacks (these will be nodes along the move tree)

Each Spot:
 - None
 - None (En Passant)
 - Pawn
 - Knight
 - Bishop
 - Rook
 - Rook (w/ castling)
 - Queen
 - King

Number of different game states:
2 (possible turns) + (8 * 8 spots * 9 possible states)
Total: 578 possibilities

Note: When castling is available, the corner pieces must be rooks so the rook is not a rook but instead a CASTLING rook. Additionally, when an empty square can be reached through en passant, it is no longer an empty square but instead an EN PASSANT empty square.

#### Implementation:
  - Based off the "DeepChess" paper

---

Gold Mind (WIP)

<br>
---

#### Environment Setup:

Run `pip install .` to install all of the required libraries

Silver Mind must be trained with PGN files. PGN stands for Portable Game Notation and is the standard format for saving 
chess games. They can be downloaded from a variety of places off the internet. Save one file per game to a folder in 
the project directory called `pgns` then run `examples/train.ipynb` to train Silver Mind.
