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


class Group(object):
    """ represents a cluster of stones that are directly touching one another """

    def __init__(self, go_board, liberties=set(), stones=set()):
        self._go_board = go_board
        self.liberties = liberties
        self.stones = stones

        for coords in self.stones:
            self._go_board.coords_to_group[coords] = self

    @staticmethod
    def merge(go_board, groups):
        liberties = set()
        stones = set()

        for group in groups:
            liberties |= group.liberties
            stones |= group.stones

        new_group = Group(go_board, liberties, stones)
        return new_group

    def add_liberties(self, liberties):
        self.liberties |= liberties

    def add_stone(self, coords):
        self.liberties.discard(coords)
        self.stones.add(coords)
        self._go_board.coords_to_group[coords] = self

    def remove_liberty(self, point):
        """
            removes the specified liberty, and deletes the group from the board if there are no liberties left
            returns the points that were captured
        """

        self.liberties.discard(point)
        if not self.liberties:


            # delete group since we have no liberties
            for coords in self.stones:
                print(self._go_board[coords])

                self._go_board[coords] = Color.UNOCCUPIED
                self._go_board.coords_to_group[coords] = None

            # update liberties of adjacent groups
            for coords in self.stones:
                for adj_coords in self._go_board.get_adjacent_coords(coords):
                    if self._go_board[adj_coords] != Color.UNOCCUPIED:
                        adj_group = self._go_board.coords_to_group[adj_coords]
                        logging.debug("returning liberties to: {}".format(adj_group))
                        adj_group.liberties.add(coords)

            return self.stones
        return set()

    def __repr__(self):
        return "liberties: {}, stones: {}".format(self.liberties, self.stones)


class GoBoard(object):
    """
        handles interaction with a go board
        e.g. placing a stone
    """

    def __init__(self, board_size):
        self._board = np.full(board_size, Color.UNOCCUPIED)
        self._ko_point = None
        self.coords_to_group = {}

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    @property
    def num_rows(self):
        return self._board.shape[0]

    @property
    def num_cols(self):
        return self._board.shape[1]

    def _is_valid_coords(self, coords):
        row_idx, col_idx = coords
        row_valid = 0 <= row_idx and row_idx < self.num_rows
        col_valid = 0 <= col_idx and col_idx < self.num_cols
        return row_valid and col_valid

    def get_adjacent_coords(self, coords):
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

    def get_diagonal_coords(self, coords):
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

    def is_safe_eye(self, coords, color):
        """
            Returns whether a specified coordinate is "safe" and should not be filled in.
        """
        adjacent_coords = self.get_adjacent_coords(coords)
        adjacent_values = self.board[adjacent_coords]

        if not np.all(adjacent_values == color):
            return False

        diagonal_coords = self.get_diagonal_coords(coords)
        diagonal_values = self.board[diagonal_coords]

        num_opposite_color = np.sum(diagonal_values == Color.get_opposite_color(color))

        # if we are in the middle, we can afford one stone of opposite color
        # otherwise, at an edge or corner, any stone of opposite color 
        # on the diagonals make a fake eye
        return (len(diagonal_values) == 8 and num_opposite_color <= 1) or num_opposite_color == 0

    def _check_move(self, coords, color):
        valid_move = False
        liberties = set()
        same_color_groups = []
        opposite_color_groups = []
        get_return_value = lambda: (valid_move, liberties, same_color_groups, opposite_color_groups)

        if self._board[coords] != Color.UNOCCUPIED:
            logging.debug("point is already occupied")
            return get_return_value()

        # find adjacent groups
        opposite_color = Color.get_opposite_color(color)

        for adj_coords in self.get_adjacent_coords(coords):
            logging.debug("adj coord: {}".format(adj_coords))

            adj_point = self._board[adj_coords]
            adj_group = self.coords_to_group.get(adj_coords)

            if adj_point == Color.UNOCCUPIED:
                logging.debug("found liberty: {}".format(adj_point))
                liberties.add(adj_coords)
                # we have a liberty, so the move is valid
                valid_move = True 

            elif adj_point == color:
                logging.debug("found same color group {}".format(adj_group))
                same_color_groups.append(adj_group)

                # the move is valid if group would have a liberty even after we place the stone
                if len(adj_group.liberties) > 1:
                    valid_move = True

            elif adj_point == opposite_color:
                logging.debug("found opposite color group {}".format(adj_group))
                opposite_color_groups.append(adj_group)

                # move is valid if we would capture a group of the oppposite color
                # that is not forbidden by the ko rule
                would_capture = len(adj_group.liberties) <= 1
                not_ko_point = len(adj_group.stones) > 1 or self._ko_point not in adj_group.stones
                if would_capture and not_ko_point:
                    valid_move = True

        return get_return_value()

    def place_stone(self, coords, color):
        """ 
            attempt to place a stone on the board
            returns True if it succeeds, and False (with no change to board state) if move is invalid
        """

        logging.debug("attempting to place {} stone at {}".format(color, coords))
        valid_move, liberties, same_color_groups, opposite_color_groups = self._check_move(coords, color)
        if not valid_move:
            return False

        # place stone and merge with adjacent groups
        self._board[coords] = color

        new_group = Group.merge(self, same_color_groups)
        new_group.add_stone(coords)
        new_group.add_liberties(liberties)

        logging.debug("created group with liberties: {} and points: {}".format(new_group.liberties, new_group.stones))

        # remove liberties from opposite color groups
        # and delete if needed
        captured_points = set()
        for group in opposite_color_groups:
            captured_points |= group.remove_liberty(coords)

        # update ko point
        if len(captured_points) == 1:
            self._ko_point = coords
        else:
            self._ko_point = None

        return True


    def get_reasonable_moves(self, color):
        """
            Returns a numpy array that is True wherever a move would be:
            1. legal 
            2. not terrible (e.g. filling in own eye).
        """
        reasonable_moves = np.zeros((self.num_rows, self.num_cols))
        for coords in np.ndindex(*reasonable_moves.shape):
            valid_move = self._check_move(coords, color)
            safe_eye = self._is_safe_eye(coords, color)
            reasonable_moves[coords] = valid_move and not safe_eye
        return reasonable_moves


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
        self._board = GoBoard(board_size)
        self._next_color = Color.BLACK
        self._num_consecutive_passes = 0 # game ends after two consecutive passes
        self._num_moves = 0

    @property
    def board(self):
        return self._board

    @property
    def next_color(self):
        return self._next_color

    def is_finished(self):
        return self._num_consecutive_passes >= 2 or self._num_moves > self.MAX_MOVES

    def place_stone(self, coords):
        if self.is_finished():
            logging.debug("game has already ended")
            return False

        valid_move = self._board.place_stone(coords, self._next_color)
        if valid_move:
            self._next_color = Color.get_opposite_color(self._next_color)
            self._num_consecutive_passes = 0
            self._num_moves += 1
        return valid_move

    def pass_turn(self):
        if self.is_finished():
            logging.debug("game has already ended")
            return
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
                for adj_coords in board.get_adjacent_coords(coords):
                    flood_fill(board, adj_coords, color, False)


        # flood fill changes the board
        # so we do it on a copy
        score_board = copy.deepcopy(self._board)
        opposite_color = Color.get_opposite_color(color)

        for row_idx in range(score_board.num_rows):
            for col_idx in range(score_board.num_cols):
                flood_fill(score_board, (row_idx, col_idx), opposite_color)

        # count the number of non flood-filled squares
        score = 0
        for row in score_board:
            for point in row:
                if point != opposite_color:
                    score += 1
        return score


    def get_scores(self):
        """ score the board & return the scores of each player, including komi """

        return {
            Color.WHITE: self._get_score(Color.WHITE) + self.KOMI,
            Color.BLACK: self._get_score(Color.BLACK),
        }


    def get_winner(self):
        scores = self.get_scores()
        keys = list(scores.keys())
        values = list(scores.values())
        return keys[np.argmax(values)]



        




