import chess
import numpy as np


class BoardState:

	def __init__(self, board=None):
		if board == None:
			self.board = chess.board
		else:
			self.board = board
	
	def __str__(self):
		return self.board
	
	def value(self):
		"""
		Returns the likelihood of black (-1) or white (1) winning given the current state
		"""
		return 1

	def serialize(self):
		"""
		Returns serialized board in format	
		 
		"""
		bstate = np.zeros(shape=64, dtype=np.uint)

		pieces_map = {"p": 1, "n":2, "b": 3, "q":4, "k":5, "r": 6}

		# General cases
		for i, piece in self.board.piece_map().items():
			piece = str(piece)
			piece_num = 0
			if piece.isupper():
				piece_num += 8
				piece = piece.lower()
			piece_num += pieces_map[piece]
			bstate[i] = piece_num

		# Special cases
		## Castling
		for location, has_rights in [(0,  self.board.has_queenside_castling_rights(True)),
								 	 (7,  self.board.has_kingside_castling_rights(True)),
								 	 (63, self.board.has_kingside_castling_rights(False)),
								 	 (63 - 7, self.board.has_queenside_castling_rights(False))]:
			if has_rights:
				bstate[location] += 1

		## En Passant
		ep_square = self.board.ep_square
		if ep_square is not None:
			bstate[ep_square] = 8

		bstate = bstate.reshape((8, 8))

		state = np.zeros(shape=(5,8,8), dtype=np.uint8)

		# Bitwise magic to convert everything into binary values
		state[0] = (bstate>>0)&1
		state[1] = (bstate>>1)&1
		state[2] = (bstate>>2)&1
		state[3] = (bstate>>3)&1

		state[4] = self.board.turn*1.0

		return state


if __name__ == "__main__":
	board = chess.Board()
	s = BoardState(board)
	print(s.serialize())

