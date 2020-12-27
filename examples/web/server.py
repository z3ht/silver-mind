from flask import Flask, request
from flask_cors import CORS


board = None
app = Flask(__name__)
CORS(app)


def computer_move(cur_board):
  ideal_move = 1 if cur_board.turn == chess.WHITE else -1
  best_move_score = -1
  best_move = None
  for move in cur_board.legal_moves:
    cur_board.push(move)
    pos_move_score = abs(ideal_move - states.BoardState(cur_board).value(valuator))
    if best_move_score < pos_move_score:
      best_move_score = pos_move_score
      best_move = move
    cur_board.pop()
  return best_move


@app.route('/chess/next', methods=["POST"])
def continue():
  comp_move = computer_move(board)
  board.push(comp_move)
  return board.fen()


@app.route('/chess/undo', methods=["POST"])
def undo():
  board.pop()
  return board.fen()


@app.route('/chess/move', methods=["POST"])
def make_move():
  move = request.args["Move"]
  if move not in board.legal_moves:
    return "false"
  board.push(move)
  return board.fen()


@app.route('/chess/start', methods=["POST"])
def start():
  board = chess.Board()


if __name__ == "__main__":
  os.environ["FLASK_ENV"] = "development"
  app.run(ssl_context="adhoc", port=8421, debug=True)
