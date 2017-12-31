from go_implementation import Color, GoGame

import logging
import pygame

BURLY_WOOD = (238, 197, 145)
BURNT_UMBER = (138, 51, 36)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Button(object):
	def __init__(
		self, coords, size, text, handler, 
		border_color=BLACK, button_color=BURLY_WOOD, font_color=BLACK, font_name='arial', 
	):
		"""
			coords: (x, y) of top left corner
			size: (height, width)
			bg_color: color of background of button
			button_color: color of text
			font_name: name of font to use
			text: text to display on button
			handler: function to call on mouse click 
		"""
		self.x1, self.y1 = coords
		self.height, self.width = size

		self.x2 = self.x1 + self.width
		self.y2 = self.y1 + self.height

		self.button_color = button_color
		self.font_color = font_color
		self.font_name = font_name
		self.text = text
		self.handler = handler

	def draw(self, surface):
		pygame.draw.rect(
			surface,
			BROWN,
			(self.x1, self.y1, self.width, self.height)
		)

	def handle_event(self, event):
		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			x, y = event.pos
			if x < self.x1 or x >= self.x2 or y < self.y1 or y >= self.y2:
				return
			self.handler()


class GoDisplay(object):
	""" class to handle display & input handling of the go board """

	def __init__(self, go_game, display_coords, display_size):
		"""
			go_game: instance of GoGame
			display_coords: (x, y) coordinates of top left corner
			display_size: (height, width)
		"""

		self.go_game = go_game
		self.grid_height, self.grid_width = display_size

		self.grid_x1, self.grid_y1 = display_coords
		self.grid_x2 = self.grid_x1 + self.grid_width
		self.grid_y2 = self.grid_y1 + self.grid_height

		self.grid_x_spacing = self.grid_width / (self.go_game.board.num_cols - 1)
		self.grid_y_spacing = self.grid_height / (self.go_game.board.num_rows - 1)

		self.stone_x_radius = self.grid_x_spacing / 3
		self.stone_y_radius = self.grid_y_spacing / 3

		self.line_width = 4

	# convert from grid to display and vice versa
	def _get_display_coords(self, grid_coords):
		grid_row, grid_col = grid_coords
		return (self.grid_x1 + grid_col * self.grid_x_spacing, self.grid_x1 + grid_row * self.grid_y_spacing)

	def _get_grid_coords(self, display_coords):
		x, y = display_coords

		col_idx = int((x + self.grid_x_spacing / 2 - self.grid_x1) / self.grid_x_spacing)
		row_idx = int((y + self.grid_y_spacing / 2 - self.grid_y1) / self.grid_y_spacing)

		row_out_of_range = row_idx < 0 or row_idx >= self.go_game.board.num_rows
		col_out_of_range = col_idx < 0 or col_idx >= self.go_game.board.num_cols
		if row_out_of_range or col_out_of_range:
			return None

		return row_idx, col_idx

	def draw(self, surface):
		""" draw the board onto the given surface """

		surface.fill(BURNT_UMBER)

		pygame.draw.rect(
			surface,
			BURLY_WOOD,
			(self.grid_x1, self.grid_y1, self.grid_width, self.grid_height),
		)

		for row_idx in range(self.go_game.board.num_rows):
			pygame.draw.line(
				surface, 
				BLACK, 
				self._get_display_coords((row_idx, 0)),
				self._get_display_coords((row_idx, self.go_game.board.num_cols - 1)),
				self.line_width,
			)

		for col_idx in range(self.go_game.board.num_cols):
			pygame.draw.line(
				surface, 
				BLACK,
				self._get_display_coords((0, col_idx)),
				self._get_display_coords((self.go_game.board.num_rows - 1, col_idx)),
				self.line_width,
			)

		for row_idx, row in enumerate(self.go_game.board):
			for col_idx, point in enumerate(row):
				if point.color == Color.UNOCCUPIED:
					continue

				x, y = self._get_display_coords((row_idx, col_idx))
				pygame.draw.ellipse(
					surface,
					BLACK if point.color == Color.BLACK else WHITE,
					(x - self.stone_x_radius, y - self.stone_y_radius, self.stone_x_radius * 2, self.stone_y_radius * 2),
				)

	def handle_event(self, event):
		""" handle user input - e.g. mouse click on board """

		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			grid_coords = self._get_grid_coords(event.pos)
			if grid_coords != None:
				self.go_game.place_stone(grid_coords)


