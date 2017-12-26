import enum
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

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
        return hash((self.x, self.y))


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
    def merge(g1, g2):
        new_group = Group(
            liberties=g1.liberties | g2.liberties,
            points=g1.points | g2.points,
        )
        return new_group

class GoBoard(object):
    # size is width x height
    def __init__(self, size=(5, 5)):
        num_rows, num_cols = size

        self._board = [
            [Point((row_idx, col_idx)) for col_idx in range(num_cols)]
            for row_idx in range(num_rows)
        ]

        self._history = []
        self._ko_point = None
        self._size = size

    # color must be one of Color.BLACK or Color.WHITE
    def place_stone(self, coords, color):
        row_idx, col_idx = coords
        point = self._board[row_idx][col_idx]

        point.color = color

    @property
    def board(self):
        return self._board





        




