

class ChessAgent:

    def __init__(self):
        self.side = None

    def reset(self, side):
        """
        Begin a new chess game and set the agent's side

        :param side: chess side ("white" or "black")
        """
        self.side = side


class ReinforcementChessAgent(ChessAgent):

    def interpret_game_state(self, game_state):
        """
        Interpret agent's observation of the game state

        :param game_state: Chess game state

        :return: Agent's observation of the game state
        """
        pass

    def interpret_action(self, raw_action):
        """
        Interpret agent's action from a raw action number

        :param raw_action: Raw action number

        :return: Formatted action
        """
        pass

    def calc_reward(self, board, action):
        """
        Calculate a reasonable reward for the agent's move

        :param board: Current game board
        :param action:
        :return:
        """
        pass


class ProceduralChessAgent(ChessAgent):

    def move(self):
        pass


class SavedSilverMindAgent(ProceduralChessAgent):
    pass


class PGNChessAgent(ProceduralChessAgent):
    pass


class ProceduralChessAgent(ProceduralChessAgent):
    pass
