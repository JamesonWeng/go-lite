from go_implementation import Color, GoGame

import logging
import pygame

BURLY_WOOD = (238, 197, 145)
BURNT_UMBER = (138, 51, 36)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class DisplayObject(object):
	def draw(self, surface):
		pass

	def handle_event(self, event):
		pass

class Box(DisplayObject):
	def __init__(self, **kwargs):
		"""
			coords: (x, y) coordinate of top left corner
			size: (height, width)
			add_border: boolean
			border_color: color of border
			border_width: width of border
			fill_color: color of box
		"""
		self.x1, self.y1 = kwargs['coords']
		self.height, self.width = kwargs['size']

		self.x2 = self.x1 + self.width
		self.y2 = self.y1 + self.height

		self._border_color = kwargs.get('border_color', BLACK)
		self._border_width = kwargs.get('border_width', 5)
		self._fill_color = kwargs.get('fill_color', BURLY_WOOD)

	def draw(self, surface):
		pygame.draw.rect(
			surface,
			self._fill_color,
			(self.x1, self.y1, self.width, self.height)
		)
		pygame.draw.rect(
			surface,
			self._border_color,
			(self.x1, self.y1, self.width, self.height),
			self._border_width,
		)


class TextBox(Box):
	def __init__(self, **kwargs):
		"""
			font_color: color of text
			font_name: name of font to use
			font_size: size of font
			text: what to write on the textbox
		"""
		super().__init__(**kwargs)

		font_name = kwargs.get('font_name', 'Comic Sans MS')
		font_size = kwargs.get('font_size', 30)
		font_color = kwargs.get('font_color', BLACK)
		font = pygame.font.SysFont(font_name, font_size)

		text = kwargs.get('text', '')
		self._text_surface = font.render(text, True, font_color)

		text_width, text_height = font.size(text)
		self._text_x1 = self.x1 + (self.width - text_width) / 2
		self._text_y1 = self.y1 + (self.height - text_height) / 2

	def draw(self, surface):
		super().draw(surface)
		surface.blit(self._text_surface, (self._text_x1, self._text_y1))


class Button(TextBox):
	def _handle(self):
		pass

	def handle_event(self, event):
		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			x, y = event.pos
			x_in_range = self.x1 <= x and x <= self.x2
			y_in_range = self.y1 <= y and y <= self.y2
			if x_in_range and y_in_range:
				self._handle()


class PassButton(Button):
	def __init__(self, *args, **kwargs):
		kwargs['text'] = 'Pass'
		super().__init__(*args, **kwargs)

		self._go_game = kwargs['go_game']

	def _handle(self):
		self._go_game.pass_turn()


class GoDisplay(Box):
	""" class to handle display & input handling of the go board """

	def __init__(self, **kwargs):
		"""
			go_game: instance of GoGame
			coords: (x, y) coordinates of top left corner
			size: (height, width)
		"""

		super().__init__(**kwargs)
		self._go_game = kwargs['go_game']

		self._x_spacing = self.width / (self._go_game.board.num_cols - 1)
		self._y_spacing = self.height / (self._go_game.board.num_rows - 1)

		self._stone_x_radius = self._x_spacing / 3
		self._stone_y_radius = self._y_spacing / 3

		self._line_width = kwargs.get('line_width', 4)

	# convert from grid to display and vice versa
	def _get_display_coords(self, grid_coords):
		grid_row, grid_col = grid_coords
		return (
			self.x1 + grid_col * self._x_spacing, 
			self.y1 + grid_row * self._y_spacing,
		)

	def _get_coords(self, display_coords):
		x, y = display_coords

		col_idx = int((x + self._x_spacing / 2 - self.x1) / self._x_spacing)
		row_idx = int((y + self._y_spacing / 2 - self.y1) / self._y_spacing)

		row_out_of_range = row_idx < 0 or row_idx >= self._go_game.board.num_rows
		col_out_of_range = col_idx < 0 or col_idx >= self._go_game.board.num_cols
		if row_out_of_range or col_out_of_range:
			return None

		return row_idx, col_idx

	def draw(self, surface):
		""" draw the board onto the given surface """

		pygame.draw.rect(
			surface,
			BURLY_WOOD,
			(self.x1, self.y1, self.width, self.height),
		)

		for row_idx in range(self._go_game.board.num_rows):
			pygame.draw.line(
				surface, 
				BLACK, 
				self._get_display_coords((row_idx, 0)),
				self._get_display_coords((row_idx, self._go_game.board.num_cols - 1)),
				self._line_width,
			)

		for col_idx in range(self._go_game.board.num_cols):
			pygame.draw.line(
				surface, 
				BLACK,
				self._get_display_coords((0, col_idx)),
				self._get_display_coords((self._go_game.board.num_rows - 1, col_idx)),
				self._line_width,
			)

		for row_idx, row in enumerate(self._go_game.board):
			for col_idx, point in enumerate(row):
				if point == Color.UNOCCUPIED:
					continue

				x, y = self._get_display_coords((row_idx, col_idx))
				pygame.draw.ellipse(
					surface,
					BLACK if point == Color.BLACK else WHITE,
					(x - self._stone_x_radius, y - self._stone_y_radius, self._stone_x_radius * 2, self._stone_y_radius * 2),
				)

	def handle_event(self, event):
		""" handle user input - e.g. mouse click on board """

		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			grid_coords = self._get_coords(event.pos)
			if grid_coords != None:
				self._go_game.place_stone(grid_coords)


