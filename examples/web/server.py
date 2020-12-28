from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import sys
import chess

sys.path.append("../..")

from silvermind import states, nnets

board = None
valuator = None
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


@app.after_request
def after(response):
  headers = response.headers
  headers["Access-Control-Allow-Origin"] = "*"
  return response


@app.route('/chess/next')
def do_computer_move():
  comp_move = computer_move(board)
  board.push(comp_move)
  return board.fen()


@app.route('/chess/undo')
def undo():
  board.pop()
  return board.fen()


@app.route('/chess/move', methods=["POST", "GET"])
def make_move():
  move = request.args["Move"]
  move = chess.Move.from_uci(move)
  if move in board.legal_moves:
    board.push(move)
  return board.fen()


@app.route('/chess/start')
def start():
  global board, valuator
  board = chess.Board()
  valuator = nnets.TwitchChess()
  valuator.load("../../models/tiny_tc.tf")
  return board.fen()


if __name__ == "__main__":
  os.environ["FLASK_ENV"] = "development"
  app.run(ssl_context="adhoc", port=8421, debug=True)
