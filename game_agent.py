"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import logging
import math
import random

MOVE_IF_NO_MOVE = (-1, -1)

# Configure logging for debugging
logging.basicConfig(filename = 'game_agent.log', level = logging.INFO)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return num_my_moves_score(game, player)

def num_my_moves_score(game, player):
    """ Heuristic 1:
    Scoring heuristic based on the number of legal moves player has
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return float(len(game.get_legal_moves(player)))

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=25.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        logging.info("Timeout set to %fms", timeout)
        self.best_move_so_far = None
        self.best_move_score = None

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        logging.debug("Time left: %f ms", time_left())
        self.time_left = time_left
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # If are no legal moves, return (-1, -1)
        if not legal_moves:
            logging.debug("No legal moves")
            return (-1, -1)

        # If this is the first move, use the opening book
        # Right now, the book is very simple - pick the center
        # TODO: Improve opening book
        if game.move_count == 1:
            logging.debug("Opening book move")
            return math.floor(game.height / 2), math.floor(game.width / 2)

        # Keep track of the best move so far
        self.best_move_so_far = None
        self.best_move_score = None

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # If we are using iterative deepening, search depth should start at 1
            if self.search_depth == -1 and self.iterative:
                self.search_depth = 1

            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise Timeout()

                if self.method == 'minimax':
                    # Run minimax
                    logging.debug("Running minimax with search depth %d", self.search_depth)
                    score, move = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    # Run alpha beta pruning
                    logging.debug("Running alpha beta pruning with search depth %d", self.search_depth)
                    score, move = self.alphabeta(game, self.search_depth)
                else:
                    logging.warning("Unknown search method: %s", self.method)

                # Keep track of best move so far
                if self.best_move_so_far is None or self.best_move_score < score:
                    self.best_move_so_far = move
                    self.best_move_score = score

                # If we need iterative deepening, proceed with that
                if not self.iterative:
                    break
                else:
                    logging.debug("Iterative deepening. Increasing search depth to %d", self.search_depth + 1)
                    self.search_depth += 1 # Iterative search with greater depth

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logging.debug("Encountered timeout")

            # If no move was selected so far, just use the first legal move
            if self.best_move_so_far is None:
                logging.debug("Timed out without a best move selected")
                self.best_move_so_far = legal_moves[0]

        # Return the best move from the last completed search iteration
        return self.best_move_so_far

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        legal_moves = game.get_legal_moves(game.active_player)

        # Terminal test - depth 0
        if depth <= 0:
            return self.score(game, game.active_player), MOVE_IF_NO_MOVE

        best_minimax_move_so_far = None
        best_minimax_move_score = None

        for legal_move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            new_game = game.forecast_move(legal_move)
            if maximizing_player:
                new_score = self.minimax_min_value(new_game, depth - 1, game.active_player)
            else:
                new_score = self.minimax_max_value(new_game, depth - 1, game.active_player)
            if best_minimax_move_so_far is None or best_minimax_move_score < new_score:
                best_minimax_move_so_far = legal_move
                best_minimax_move_score = new_score
        return best_minimax_move_score, best_minimax_move_so_far

    def minimax_max_value(self, game, depth, player):
        """Implement max-value (Russell & Norvig) and proceed to next depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The score for the current search branch

        """
        # Terminal test - depth 0
        if depth <= 0:
            return self.score(game, player)

        v = float("-inf")
        for legal_move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            v = max(v, self.minimax_min_value(game.forecast_move(legal_move), depth - 1, player))
        return v

    def minimax_min_value(self, game, depth, player):
        """Implement min-value (Russell & Norvig) and proceed to next depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The score for the current search branch

        """
        # Terminal test - depth 1
        if depth <= 0:
            return self.score(game, player)

        v = float("inf")
        for legal_move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            v = min(v, self.minimax_max_value(game.forecast_move(legal_move), depth - 1, player))
        return v

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if maximizing_player:
            v, move = self.alphabeta_max_value(game, depth, alpha, beta, game.active_player)
        else:
            v, move = self.alphabeta_min_value(game, depth, alpha, beta, game.active_player)
        return v, move

    def alphabeta_max_value(self, game, depth, alpha, beta, player):
        """Implement max-value (Russell & Norvig) and proceed to next depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        # Terminal test - depth 0
        if depth <= 0:
            return self.score(game, player), MOVE_IF_NO_MOVE

        v = float("-inf")
        best_move = MOVE_IF_NO_MOVE # Keep track of the best move
        for legal_move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            min_value, min_move = self.alphabeta_min_value(game.forecast_move(legal_move), depth - 1, alpha, beta, player)
            if min_value > v: # v = max(v, min_value)
                v = min_value
                best_move = legal_move
            if v >= beta:
                return v, legal_move
            alpha = max(alpha, v)
        return v, best_move

    def alphabeta_min_value(self, game, depth, alpha, beta, player):
        """Implement min-value (Russell & Norvig) and proceed to next depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        # Terminal test - depth 0
        if depth <= 0:
            return self.score(game, player), MOVE_IF_NO_MOVE

        v = float("inf")
        best_move = MOVE_IF_NO_MOVE # Keep track of the best move
        for legal_move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            max_value, max_move = self.alphabeta_max_value(game.forecast_move(legal_move), depth - 1, alpha, beta, player)
            if max_value < v: # v = min(v, max_value)
                v = max_value
                best_move = legal_move
            if v <= alpha:
                return v, legal_move
            beta = min(beta, v)
        return v, best_move


