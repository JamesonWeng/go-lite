from go_board import Color, GoBoard
import pygame



# class to handle drawing / user input for the go board
class GoDisplay(object):
	BLACK = (0, 0, 0)
	BROWN = (165, 42, 42)
	WHITE = (255, 255, 255)

	def __init__(self, num_rows, num_cols, grid_height, grid_width, grid_x1, grid_y1):
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.board = GoBoard((self.num_rows, self.num_cols))

		self.grid_height = grid_height
		self.grid_width = grid_width

		self.grid_x1 = grid_x1
		self.grid_y1 = grid_y1
		self.grid_x2 = self.grid_x1 + self.grid_width
		self.grid_y2 = self.grid_y1 + self.grid_height

		self.grid_x_spacing = self.grid_width / (self.num_cols - 1)
		self.grid_y_spacing = self.grid_height / (self.num_rows - 1)

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

		if row_idx < 0 or row_idx >= self.num_rows or col_idx < 0 or col_idx >= self.num_cols:
			return None

		return row_idx, col_idx

	def draw(self, surface):
		surface.fill(self.BROWN)

		for row_idx in range(self.num_rows):
			pygame.draw.line(
				surface, 
				self.BLACK, 
				self._get_display_coords((row_idx, 0)),
				self._get_display_coords((row_idx, self.num_cols - 1)),
				self.line_width,
			)

		for col_idx in range(self.num_cols):
			pygame.draw.line(
				surface, 
				self.BLACK,
				self._get_display_coords((0, col_idx)),
				self._get_display_coords((self.num_rows - 1, col_idx)),
				self.line_width,
			)

		for row_idx, row in enumerate(self.board):
			for col_idx, point in enumerate(row):
				if point.color == Color.UNOCCUPIED:
					continue

				x, y = self._get_display_coords((row_idx, col_idx))
				pygame.draw.ellipse(
					surface,
					self.BLACK if point.color == Color.BLACK else self.WHITE,
					(x - self.stone_x_radius, y - self.stone_y_radius, self.stone_x_radius * 2, self.stone_y_radius * 2),
				)

	def handle_event(self, event):
		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			grid_coords = self._get_grid_coords(event.pos)
			if grid_coords != None:
				self.board.place_stone(grid_coords)
				return True
		return False