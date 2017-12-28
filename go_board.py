import enum
import logging
import numpy as np
import os

class Color(enum.Enum):
    BLACK = enum.auto()
    UNOCCUPIED = enum.auto()
    WHITE = enum.auto()

class Point(object):
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
        return '({}, {}, {})'.format(self._row_idx, self._col_idx, self._color)


class Group(object):
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

    # removes the specified liberty
    # and deletes the pieces from the board if no liberties left
    # returns the points that were captured
    def remove_liberty(self, board, point):
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
                        logging.debug('returning liberties to: {}'.format(adj_p.group))
                        adj_p.group._liberties.add(p)
            return self._points
        return set()

    def __repr__(self):
        return 'liberties: {}, points: {}'.format(self._liberties, self._points)


class GoBoard(object):
    # size is width x height
    def __init__(self, size=(5, 5)):
        self._num_rows, self._num_cols = size

        self._board = [
            [Point((row_idx, col_idx)) for col_idx in range(self._num_cols)]
            for row_idx in range(self._num_rows)
        ]

        self._history = []
        self._ko_point = None
        self._next_color = Color.BLACK
        self._size = size

    @staticmethod
    def _get_opposite_color(color):
        return Color.BLACK if color == Color.WHITE else Color.WHITE

    def get_adjacent_points(self, point):
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

    # attempt to place a stone at the given coordinates
    # returns False if this is not a valid move
    def place_stone(self, coords):
        logging.debug('trying to place {} stone at {}'.format(self._next_color, coords))
        row_idx, col_idx = coords

        target_point = self._board[row_idx][col_idx]
        if target_point.color != Color.UNOCCUPIED:
            logging.debug('point is already occupied')
            return False

        # find adjacent groups
        opposite_color = self._get_opposite_color(self._next_color)
        same_color_groups = []
        opposite_color_groups = []
        liberties = set()
        valid_move = False

        for adj_point in self.get_adjacent_points(target_point):
            logging.debug('adj point: {}'.format(adj_point))

            if adj_point.color == Color.UNOCCUPIED:
                logging.debug('found liberty: {}'.format(adj_point))
                liberties.add(adj_point)

                # we have a liberty, so the move is valid
                valid_move = True

            elif adj_point.color == self._next_color:
                logging.debug('found same color group {}'.format(adj_point.group))
                same_color_groups.append(adj_point.group)

                # the move is valid if group would have a liberty even after we place the stone
                if len(adj_point.group.liberties) > 1:
                    valid_move = True

            elif adj_point.color == opposite_color:
                logging.debug('found opposite color group {}'.format(adj_point.group))

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
        target_point.color = self._next_color
        new_group.add_stone(target_point)
        new_group.add_liberties(liberties)

        logging.debug('created group with liberties: {} and points: {}'.format(new_group.liberties, new_group.points))

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
        
        # set next color
        self._next_color = opposite_color
        return True

    def __getitem__(self, key):
        return self._board[key]



        




