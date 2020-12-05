### RESEARCH:

#### Day 1:

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

#### Day 2:
Yesterday I got pretty distracted with trying to find a good model. Today I'm going to focus more on creating a really basic (shit) model and improving from there. I'm going to use PyTorch and numpy. I'm using games from this chess database http://rebel13.nl/rebel13/rebel%2013.html. I'm not sure if they'll be any good but it has 2.5 million games.



