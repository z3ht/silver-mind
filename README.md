# Silver Mind  
**Reinforcement learning chess AI**  
Designed By: Andrew Darling

![404](https://i.imgur.com/f0RThee.jpg)  
--- *GangstaRaptor by: arbsbuhpuh (https://imgur.com/gallery/d6vxE1f)* ---

<br>

**Goal:**  
The goal for creating Silver Mind is to build a Chess AI using reinforcement learning that can beat me 
(~1025 elo) at chess. A stretch goal is to beat Magnus Carlson at age 18 from the Magnus Carlson app having 
only trained on games from Garry Kasparov and other GMs. This would be a great accomplishment because Magnus Carlson 
beat Kasparov at age 16 in 2004 and has only improved since then so Silver Mind would be required to learn a 
significant amount beyond its training.

**Strategy:**  
Develop two algorithms: Silver Mind and Gold Mind. Silver Mind will be built from training data and should know possible moves it has available. Gold Mind should be capable of making any move (even illegal ones) but learn which are possible and more so learn which ones will be strong

Silver Mind:
1) Generate tree of possible moves
2) Develop an understanding of high and low win percentage states
    - Minimax: Move to the best possible spot if your opponent makes the best moves possible (respond to worst possible outcome)
    	- This will probably be difficult because following a move tree in chess will take forever
    - Bellman Equation: Realize reward as the value of being in the current spot + loss value * being in the next spot
	Representation:
		- Winning State: 1
		- Draw: 0
		- Losing State: -1
	     	- F(one_state_away_from_winning) =  0 (reward for being at current position) + epsilon (loss function) * 1 (F(next_best_state which is winning state))
3) Move to the best possible state from the available tree of moves
3) Continue until victory

Definitions:

Chess State:
A few options - `https://en.wikipedia.org/wiki/Board_representation_(computer_chess)`

Things to represent:
 - 8x8 Board: Data about each spot (8 possible states)
 - Turn
 - White Kingside castle
 - White Queenside castle
 - Black Kingside castle
 - Black Queenside castle 

Things to ignore:
 - 50-move draw rule (I will not factor this into my model)
 - 3 move surrender (I will not factor this into my model)
 - Possible attacks (these will be nodes along the move tree)

Each Spot:
 - None
 - Pawn
 - Pawn (En Passant)
 - Knight
 - Bishop
 - Rook
 - Queen
 - King

Do I need to store en passant or can this just be a node in the move tree?
	- I do need to keep it because a board can look like en passant is available even when its not

Do I need to store castling info (or can it be a move in the node tree until lost)?
	- I do need to keep it because a board can look like castling is available without castling actually being available

Do I need to store whose turn it is?
	- Yes same reason as above two

Do I need to store each colors castling availability?
	- Yes alternating the state depending on whose turn it is seems like a pain and might not be possible

Number of different game states:
2 (possible turns) + 2 (each color) * (2 (kingside castle availability) + 2 (queenside availability)) * (8 * 8 spots * 8 possible states)
Total: 2 + 2 * 4 + 8 * 8 * 8
Total: 10 + 8^3

Silver Mind Assumptions/Advantages:
	- Provide Silver Mind with possible moves
	- Provide Silver Mind with LOTS of training data

Training Silver Mind:
	- Load lots of game data
	- X = games as list of states (in increasing order?), Y = winner
	- Incentivize wins and penalize losses

Should I remove games that end in surrender from the training data?
	- No because if someone is willing to surrender Silver Mind will have great positioning

RESEARCH:

What NN can I use?
	- LSTMs are a good candidate because they maintain state

What are LSTMs? Should I consider using transformers instead? What do I need to know about maintaining state?  
	LSTMs Crash Course: https://medium.com/@jilvanpinheiro/crash-course-in-lstm-networks-fbd242231873
		- LSTMs are an expansion upon recurrent neural networks (networks with loops in them)
		- Allow for making predictions from prior information
		- RNNs perform well when the gap between learning state and making predictions with said state is small
		- LSTMs default behavior is learning info for long periods of time
		- Essentially all achievments of RNNs are achieved by LSTMs
	Transformers Introduction: https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
		- Transformers are particularly effective with NLP
		- Transformers are much simpler than LSTMs which connect the encoder and decoder through an attention mechanism
		- Transformers work soley based off attention mechanisms
		- Require less time to train and outperform LSTMs
			- Still requires lots of training: eight gpus and 3.5 days (2017)
		Sequence-to-Sequence Architecture:
			- Transforms sequences to new sequences (i.e. sequence of words in a sentence)
			- Consist of Encoder and Decoder
				- Encoder takes input sequence and maps it to a higher dimensional space (this seems similar to SVMs)
				- Decoder takes the "abstract vector" and turns it into the output sequence
				- LSTMs can act as both the encoder and decoder which learn this imaginary higher dimensional space where they can communicate with eachother
		Attention:
			- Looks at an input sequence and decides which other parts of the sequence are important
			- Essentially, the encoder provides the decoder with key words in the encoder's "language" which makes it easier for the decoder to know what to focus on in the translation
		** Paper for good transformer architectures: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
	        ** Annotated "Attention is All you Need" paper on Transformers: http://nlp.seas.harvard.edu/2018/04/03/attention.html
	Reddit Post "Are Transformers strictly better than LSTMs?"
		- Transformers are usually better but need to be deeper/bigger
		- LSTMs are good for decoding a lot during training
	Quora Post "Benefits of Transformers over LSTMs": Transformers use attention to see everything at once. LSTMS must be processed sequentially. Therefore, LSTMs must propogate the error back in time. Transformers see all at once so there is no need for this. LSTMs do have one advantage over attention based models: They have no required input length. XLNet combines the benefits of attention based models with the recurrence mechanism of sequence models like LSTMs to eliminate the fixed size input limitation. XLNet uses segment level recurrence rather than "word" based recurrence (striking a middle ground between LSTMs and attention based models). 

What is XLNet? Generalized autoregressive pretraining method
	- AR Language Model: Uses context words to predict next words in the forward OR backwards directions (not both)
	- GPTs are AR language models (meaning they must move in the forward or backwards directions)
	- AR Language Models are strong at generative NLP tasks (this sounds like the Chess problem)
	- BERT (attention based model) is categorized as an Autoencoder Language Model:
		- Reconstructs original data from corrupted input in BOTH directions
		- This is a problem when working with live data which only moves in the forward direction
	- XLNet proposes a new way to let the AR model learn from bi-directional context
		- New Objective Function: Permutation Language Modeling (uses permutation to make predictions from data before and after target [this seems really clever])
	Original XLNet paper: https://arxiv.org/abs/1906.08237

XLNet Python module: `https://huggingface.co/transformers/model_doc/xlnet.html`

Transformers seem to be good at matching input (such as language) with similarly structured output (such as a different language). I am concerned predicting the best chess move given previous moves (and future moves in the training data) will not fit into this paradigm. Also it looks like XLNet is extremely difficult to train. These two hurdles may prevent the use of XLNet or Attention based models in general.

---

Gold Mind:
1) Reinforcement learning algorithm that competes against silver mind
2) Develop an understanding of good and bad moves
	- I will need to do a lot of research to make this happen
	- I assume I will need to develop an understanding of the value of everything on the board
	- I will want to maximize my position and minimize my opponent's 
1) Make the best possible moves

<br>
---

#### Environment Setup:

Run `pip install .` to install all of the required libraries

Silver Mind must be trained with PGN files. PGN stands for Portable Game Notation and is the standard format for saving 
chess games. They can be downloaded from a variety of places off the internet. Save one file per game to a folder in 
the project directory called `pgns` then run `examples/train.ipynb` to train Silver Mind.
