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


class Point(object):
    """
        represents a point on the board, which can be either occupied or unoccupied
        if occupied, it belongs to a group of stones
    """

    def __init__(self, coords):
        self._row_idx, self._col_idx = coords

        self._group = None
        self._color = Color.UNOCCUPIED

    @property
    def col_idx(self):
        return self._col_idx

    @property 
    def row_idx(self):
        return self._row_idx

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, val):
        self._group = val

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, val):
        self._color = val

    def __hash__(self):
        return hash((self._row_idx, self._col_idx))

    def __repr__(self):
        return "({}, {}, {})".format(self._row_idx, self._col_idx, self._color)


class Group(object):
    """ represents a cluster of stones that are directly touching one another """

    def __init__(self, liberties=set(), points=set()):
        self._liberties = liberties
        self._points = points
        for p in points:
            p.group = self

    @property
    def liberties(self):
        return self._liberties

    @property
    def points(self):
        return self._points

    @staticmethod
    def merge(groups):
        liberties = set()
        points = set()

        for group in groups:
            liberties |= group.liberties
            points |= group.points

        new_group = Group(liberties, points)
        return new_group

    def add_stone(self, point):
        self._points.add(point)
        self._liberties.discard(point)
        point.group = self

    def add_liberties(self, liberties):
        self._liberties |= liberties

    def remove_liberty(self, board, point):
        """
            removes the specified liberty, and deletes the group from the board if there are no liberties left
            returns the points that were captured
        """

        self._liberties.discard(point)
        if not self._liberties:
            # delete group since we have no liberties
            for p in self._points:
                p.color = Color.UNOCCUPIED
                p.group = None
            # update liberties of adjacent groups
            for p in self._points:
                for adj_p in board.get_adjacent_points(p):
                    if adj_p.color != Color.UNOCCUPIED:
                        logging.debug("returning liberties to: {}".format(adj_p.group))
                        adj_p.group._liberties.add(p)
            return self._points
        return set()

    def __repr__(self):
        return "liberties: {}, points: {}".format(self._liberties, self._points)


class GoBoard(object):
    """
        handles interaction with a go board
        e.g. placing a stone
    """

    def __init__(self, board_size):
        self._num_rows, self._num_cols = board_size
        self._board = [
            [Point((row_idx, col_idx)) for col_idx in range(self._num_cols)]
            for row_idx in range(self._num_rows)
        ]
        self._ko_point = None

    def __getitem__(self, key):
        return self._board[key]

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_cols(self):
        return self._num_cols

    def get_adjacent_points(self, point):
        """
            get a list of points on the board that are adjacent to the given point,
            excluding diagonally adjacent points
        """

        adj_coords = [
            (point.row_idx - 1, point.col_idx),
            (point.row_idx, point.col_idx - 1),
            (point.row_idx + 1, point.col_idx),
            (point.row_idx, point.col_idx + 1),
        ]
        adj_points = [
            self._board[row_idx][col_idx]
            for row_idx, col_idx in adj_coords
            if 0 <= row_idx and row_idx < self._num_rows and 0 <= col_idx and col_idx < self._num_cols
        ]
        return adj_points


    def place_stone(self, coords, color):
        """ 
            attempt to place a stone on the board
            returns True if it succeeds, and False (with no change to board state) if move is invalid
        """

        logging.debug("attempting to place {} stone at {}".format(color, coords))
        row_idx, col_idx = coords

        target_point = self._board[row_idx][col_idx]
        if target_point.color != Color.UNOCCUPIED:
            logging.debug("point is already occupied")
            return False

        # find adjacent groups
        opposite_color = Color.get_opposite_color(color)
        same_color_groups = []
        opposite_color_groups = []
        liberties = set()
        valid_move = False

        for adj_point in self.get_adjacent_points(target_point):
            logging.debug("adj point: {}".format(adj_point))

            if adj_point.color == Color.UNOCCUPIED:
                logging.debug("found liberty: {}".format(adj_point))
                liberties.add(adj_point)

                # we have a liberty, so the move is valid
                valid_move = True

            elif adj_point.color == color:
                logging.debug("found same color group {}".format(adj_point.group))
                same_color_groups.append(adj_point.group)

                # the move is valid if group would have a liberty even after we place the stone
                if len(adj_point.group.liberties) > 1:
                    valid_move = True

            elif adj_point.color == opposite_color:
                logging.debug("found opposite color group {}".format(adj_point.group))
                opposite_color_groups.append(adj_point.group)

                # move is valid if we would capture a group of the oppposite color
                # that is not forbidden by the ko rule
                adj_group = adj_point.group
                would_capture = len(adj_group.liberties) <= 1
                not_ko_point = len(adj_group.points) > 1 or self._ko_point not in adj_group.points
                if would_capture and not_ko_point:
                    valid_move = True

        if not valid_move:
            return False

        # place stone and merge with adjacent groups
        new_group = Group.merge(same_color_groups)
        target_point.color = color
        new_group.add_stone(target_point)
        new_group.add_liberties(liberties)

        logging.debug("created group with liberties: {} and points: {}".format(new_group.liberties, new_group.points))

        # remove liberties from opposite color groups
        # and delete if needed
        captured_points = set()
        for group in opposite_color_groups:
            captured_points |= group.remove_liberty(self, target_point)

        # update ko point
        if len(captured_points) == 1:
            self._ko_point = target_point
        else:
            self._ko_point = None

        return True


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

    def place_stone(self, coords):
        valid_move = self._board.place_stone(coords, self._next_color)
        if valid_move:
            self._next_color = Color.get_opposite_color(self._next_color)
            self._num_consecutive_passes = 0
            self._num_moves += 1
        return valid_move

    def pass_turn(self):
        self._num_consecutive_passes += 1

    def is_finished(self):
        return self._num_consecutive_passes >= 2 or self._num_moves > self.MAX_MOVES

    def _get_score(self, color):
        """
            Under Tromp-Taylor scoring, the score is the combined total of:
            - the number of stones the player has on the board
            - the number of empty points that reach only the player's stones
        """

        # we calculate the points by floodfilling the board starting from stones of opposite color
        # the non-filled squares are the points that the current color has

        def flood_fill(board, point, color):
            # for efficiency, we stop once we reach a colored point
            if point.color == Color.UNOCCUPIED: 
                point.color = color
                for adj_point in board.get_adjacent_points(point):
                    flood_fill(board, adj_point, color)

        # flood fill changes the board
        # so we do it on a copy
        score_board = copy.deepcopy(self._board)
        opposite_color = Color.get_opposite_color(color)
        for row in score_board:
            for point in row:
                # flood fill adjacent squares if it's the opposite color
                if point.color == opposite_color:
                    for adj_point in score_board.get_adjacent_points(point):
                        flood_fill(score_board, adj_point, opposite_color)

        # count the number of non flood-filled squares
        score = 0
        for row in score_board:
            for point in row:
                if point.color != opposite_color:
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
        return scores.keys()[np.argmax(scores.values())]



        




