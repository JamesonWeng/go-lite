import copy
import enum
import logging
import numpy as np
import os


class Color(enum.Enum):
    """ the color of each point on the board """

    BLACK = enum.auto()
    UNOCCUPIED = enum.auto()
    WHITE = enum.auto()

    @staticmethod
    def get_opposite_color(color):
        if color == Color.BLACK:
            return Color.WHITE
        elif color == Color.WHITE:
            return Color.BLACK
        else:
            raise Exception("Bad color")


class GoGame(object):
    """ 
        Encapsulates rules & scoring for playing a game of go using Tromp-Taylor rules, with komi.
        Currently does not have superko detection; only ko detection implemented.
    """
    
    KOMI = 6.5  # points awarded to white for starting second
    MAX_MOVES = 5000 # prevent cycles since superko is not yet implemented

    def __init__(self, board_size=(5, 5)):
        """
            board_size: (num_rows, num_cols)
        """
        self.num_rows, self.num_cols = board_size
        self._board = np.full(board_size, Color.UNOCCUPIED)
        self._ko_point = None

        self._next_color = Color.BLACK
        self._num_consecutive_passes = 0 # game ends after two consecutive passes
        self._num_moves = 0

    @property
    def board(self):
        return self._board

    @property
    def next_color(self):
        return self._next_color

    @property
    def num_moves(self):
        return self._num_moves

    def is_finished(self):
        return self._num_consecutive_passes >= 2 or self._num_moves > self.MAX_MOVES

    def _is_valid_coords(self, coords):
        row_idx, col_idx = coords
        row_valid = 0 <= row_idx and row_idx < self.num_rows
        col_valid = 0 <= col_idx and col_idx < self.num_cols
        return row_valid and col_valid

    def _get_adjacent_coords(self, coords):
        """
            get a list of coords on the board that are directly adjacent to the given coords,
            (excludes diagonally adjacent coords)
        """
        row_idx, col_idx = coords
        adj_coords = [
            (row_idx - 1, col_idx),
            (row_idx, col_idx - 1),
            (row_idx + 1, col_idx),
            (row_idx, col_idx + 1),
        ]
        adj_coords = [
            coords for coords in adj_coords
            if self._is_valid_coords(coords)
        ]
        return adj_coords

    def _get_group(self, coords):
        """
            returns stones & liberties of the group that containing coords
        """
        stones = set()
        liberties = set()
        color = self._board[coords]
        coords_to_check = [coords]

        if color == Color.UNOCCUPIED:
            return stones, liberties

        while coords_to_check:
            coords = coords_to_check.pop()
            if coords in stones: # prevent flood fill from going backwards
                continue

            if self._board[coords] == Color.UNOCCUPIED:
                liberties.add(coords)
            elif self._board[coords] == color:
                stones.add(coords)
                coords_to_check += self._get_adjacent_coords(coords)

        return stones, liberties

    def _get_diagonal_coords(self, coords):
        row_idx, col_idx = coords
        diagonal_coords = [
            (row_idx - 1, col_idx - 1),
            (row_idx - 1, col_idx + 1),
            (row_idx + 1, col_idx - 1),
            (row_idx + 1, col_idx + 1),
        ]
        diagonal_coords = [
            coords for coords in diagonal_coords
            if self._is_valid_coords(coords)
        ]
        return diagonal_coords

    def _is_safe_eye(self, coords, color):
        """
            Returns whether a specified coordinate is "safe" and should not be filled in.
        """
        adjacent_coords = self._get_adjacent_coords(coords)
        rows, cols = zip(*adjacent_coords)
        adjacent_values = self.board[rows, cols]

        if not np.all(adjacent_values == color):
            return False

        diagonal_coords = self._get_diagonal_coords(coords)
        diagonal_values = self.board[diagonal_coords]

        num_opposite_color = np.sum(diagonal_values == Color.get_opposite_color(color))

        # if we are in the middle, we can afford one stone of opposite color
        # otherwise, at an edge or corner, any stone of opposite color 
        # on the diagonals make a fake eye
        return (len(diagonal_values) == 8 and num_opposite_color <= 1) or num_opposite_color == 0

    def _check_move(self, coords, color):
        """
            Returns (valid_move, opposite_color_groups)
            valid_move: True iff placing a stone at coords is legal
            opposite_color_groups: if move is legal, this is a list of adjacent groups of opposite color
        """
        valid_move = False
        opposite_color_groups = []

        if self._board[coords] != Color.UNOCCUPIED:
            logging.debug("point is already occupied")
            return valid_move, opposite_color_groups

        # find adjacent groups
        opposite_color = Color.get_opposite_color(color)
        adj_coords_list = self._get_adjacent_coords(coords)

        while adj_coords_list:
            adj_coords = adj_coords_list.pop()
            logging.debug("processing adj coords: {}".format(adj_coords))

            adj_point = self._board[adj_coords]
            adj_group_stones, adj_group_liberties = self._get_group(adj_coords)

            # prevent duplicate groups
            adj_coords_list = [c for c in adj_coords_list if c not in adj_group_stones]

            if adj_point == Color.UNOCCUPIED:
                logging.debug("found liberty")
                valid_move = True # we have a liberty, so this move is legal

            elif adj_point == color:
                logging.debug("found same color group")

                # the move is valid if group would have a liberty even after we place the stone
                if len(adj_group_liberties) > 1:
                    valid_move = True

            elif adj_point == opposite_color:
                logging.debug("found opposite color group")
                opposite_color_groups.append((adj_group_stones, adj_group_liberties))

                # move is valid if we would capture a group of the oppposite color
                # that is not forbidden by the ko rule
                would_capture = len(adj_group_liberties) <= 1
                not_ko_point = len(adj_group_stones) > 1 or self._ko_point not in adj_group_stones
                if would_capture and not_ko_point:
                    valid_move = True

        return valid_move, opposite_color_groups

    def _place_stone(self, coords, color):
        logging.debug("attempting to place {} stone at {}".format(color, coords))
        valid_move, opposite_color_groups = self._check_move(coords, color)
        if not valid_move:
            return False

        # place stone and merge with adjacent groups
        self._board[coords] = color

        # remove opposite color groups that are out of liberties
        num_captured = 0
        for stones, liberties in opposite_color_groups:
            if len(liberties) <= 1:
                rows, cols = zip(*list(stones))
                self._board[rows, cols] = Color.UNOCCUPIED
                num_captured += len(stones)

        if num_captured == 1:
            self._ko_point = coords
        else:
            self._ko_point = None

        return True

    def place_stone(self, coords):
        """
            attempt to place a stone on the board
            returns True if it succeeds, and False (with no change to board state) if move is invalid
        """
        if self.is_finished():
            logging.debug("game has already ended")
            return False

        valid_move = self._place_stone(coords, self._next_color)
        if valid_move:
            self._next_color = Color.get_opposite_color(self._next_color)
            self._num_consecutive_passes = 0
            self._num_moves += 1
        return valid_move

    def get_reasonable_moves(self):
        """
            Returns a numpy array that is True wherever a move would be:
            - legal 
            - not terrible (e.g. filling in our own eye).
        """
        reasonable_moves = np.zeros_like(self._board)
        for coords in np.ndindex(*reasonable_moves.shape):
            valid_move, _ = self._check_move(coords, self._next_color)
            safe_eye = self._is_safe_eye(coords, self._next_color)
            reasonable_moves[coords] = valid_move and not safe_eye
        return reasonable_moves

    def pass_turn(self):
        if self.is_finished():
            logging.debug("game has already ended")
            return

        self._next_color = Color.get_opposite_color(self._next_color)
        self._num_consecutive_passes += 1

    def _get_score(self, color):
        """
            Under Tromp-Taylor scoring, the score is the combined total of:
            - the number of stones the player has on the board
            - the number of empty points that reach only the player's stones
        """
        if self._num_moves == 0:
            return 0

        # we calculate the points by floodfilling the board starting from stones of opposite color
        # the non-filled squares are the points that the current color has

        def flood_fill(board, coords, color, first_call=True):
            # for efficiency, we stop once we reach a colored point
            # on the first call, we are allowed to start from a point of the desired color
            should_flood = (
                (not first_call and board[coords] == Color.UNOCCUPIED) or 
                (first_call and board[coords] == color)
            )
            if should_flood:
                board[coords] = color
                for adj_coords in self._get_adjacent_coords(coords):
                    flood_fill(board, adj_coords, color, False)

        # flood fill changes the board
        # so we do it on a copy
        score_board = copy.deepcopy(self._board)
        opposite_color = Color.get_opposite_color(color)
        for coords in np.ndindex(*score_board.shape):
            flood_fill(score_board, coords, opposite_color)

        # count the number of non flood-filled squares
        score = np.sum(score_board != opposite_color)
        return score

    def get_scores(self):
        """ score the board & return the scores of each player, including komi """
        return {
            Color.WHITE: self._get_score(Color.WHITE) + self.KOMI,
            Color.BLACK: self._get_score(Color.BLACK),
        }

    def get_winner(self):
        """ return the winning color """
        scores = self.get_scores()
        keys = list(scores.keys())
        values = list(scores.values())
        return keys[np.argmax(values)]



        




