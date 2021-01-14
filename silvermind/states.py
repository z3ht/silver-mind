import sys
import numpy as np
from nnets import ValueNet, ComparisonNet


def value(board, chess_net, depth=0):
    """
    Get the value of the current position denoting likelihood of opposite (-) or current (+) player winning given
    the current state through the use of an Alpha-Beta pruned tree search with depth = `depth`

    Parameters
    ====
    :param board: Chess board being evaluated
    :param chess_net: The chess net that should be used for predictions
    :param depth: The depth of the alpha-beta pruned tree search
    """
    mover_color = board.turn

    def depth_score(board_vertex=board.copy(), inner_depth=depth, alpha_score=None, alpha_val=board.copy()):
        """
        Get value from Alpha-Beta Pruning search

        Global Arguments:
            - chess_net = Chess net evaluating moves
            - mover_color = The head board vertex color

        Parameters:
        ====
        :param board_vertex : The current board vertex
        :param inner_depth : Current search depth
        :param alpha_score : Current best move value
        :param alpha_val : Current best board
        """
        sign = 1 if mover_color == board_vertex.turn else -1

        if inner_depth == 0 or board_vertex.is_game_over():
            if isinstance(chess_net, ValueNet):
                return chess_net.predict(serialize(board_vertex)) * sign
            elif isinstance(chess_net, ComparisonNet):
                return chess_net.predict(serialize(board_vertex), serialize(alpha_val)) * sign
            else:
                raise TypeError(f"`chess_net` type ({type(chess_net)}) not supported")

        if mover_color == board_vertex.turn:
            if alpha_score is None:
                alpha_score = -sys.maxsize
            for move in board_vertex.legal_moves:
                board_vertex.push(move)
                alpha_val.push(move)
                cur_score = depth_score(
                    board_vertex=board_vertex, inner_depth=inner_depth - 1,
                    alpha_score=alpha_score, alpha_val=alpha_val
                ) * sign    # sign always equals 1 when mover_color == board_vertex.turn
                board_vertex.pop(move)
                if cur_score > alpha_score:
                    alpha_score = cur_score
                else:
                    alpha_val.pop(move)
        else:       # mover_color != board_vertex.turn
            if alpha_score is None:
                alpha_score = sys.maxsize
            for move in board_vertex.legal_moves:
                board_vertex.push(move)
                alpha_val.push(move)
                cur_score = depth_score(
                    board_vertex=board_vertex, inner_depth=inner_depth - 1,
                    alpha_score=alpha_score, alpha_val=alpha_val
                ) * sign    # sign always equals -1 when mover_color != board_vertex.turn
                board_vertex.pop(move)
                if cur_score >= alpha_score:
                    alpha_val.pop(move)
                    break
                else:
                    alpha_score = cur_score
        return alpha_score
    return depth_score()


def serialize(board):
    """
    Returns serialized chess board object as numpy array
    """
    bstate = np.zeros(shape=64, dtype=np.uint)

    pieces_map = {"p": 1, "n": 2, "b": 3, "q": 4, "k": 5, "r": 6}

    # General cases
    for i, piece in board.piece_map().items():
        piece = str(piece)
        piece_num = 0
        if piece.isupper():
            piece_num += 8
            piece = piece.lower()
        piece_num += pieces_map[piece]
        bstate[i] = piece_num

    # Special cases...
    # Castling
    for location, has_rights in [(0, board.has_queenside_castling_rights(True)),
                                 (7, board.has_kingside_castling_rights(True)),
                                 (63, board.has_kingside_castling_rights(False)),
                                 (63 - 7, board.has_queenside_castling_rights(False))]:
        if has_rights:
            bstate[location] += 1

    # En Passant
    ep_square = board.ep_square
    if ep_square is not None:
        bstate[ep_square] = 8

    bstate = bstate.reshape((8, 8))

    state = np.zeros(shape=(5, 8, 8), dtype=np.uint8)

    # Bitwise magic to convert everything into binary values
    state[0] = (bstate >> 0) & 1
    state[1] = (bstate >> 1) & 1
    state[2] = (bstate >> 2) & 1
    state[3] = (bstate >> 3) & 1

    state[4] = board.turn * 1.0

    return state

