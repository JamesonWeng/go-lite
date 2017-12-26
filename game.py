from go_board import Color, GoBoard

import logging
import pygame
import sys
import time

# display constants
GRID_NUM_ROWS = 5
GRID_NUM_COLUMNS = 10
GRID_SPACING = 50
GRID_HEIGHT = (GRID_NUM_ROWS - 1) * GRID_SPACING 
GRID_WIDTH = (GRID_NUM_COLUMNS - 1) * GRID_SPACING 

GRID_BORDER = 50

STONE_RADIUS = GRID_SPACING / 3

DISPLAY_HEIGHT = (GRID_NUM_ROWS - 1) * GRID_SPACING + GRID_BORDER * 2
DISPLAY_WIDTH = (GRID_NUM_COLUMNS  - 1) * GRID_SPACING + GRID_BORDER * 2

LINE_WIDTH = 4

# logging
logging.basicConfig(level=logging.DEBUG)

# set up the window
pygame.init()
display_surface = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Go Lite')

# instantiate the go board
board = GoBoard((GRID_NUM_ROWS, GRID_NUM_COLUMNS))

# convert from grid to display and vice versa
def get_display_coords(grid_coords):
	grid_row, grid_col = grid_coords
	return (GRID_BORDER + grid_col * GRID_SPACING, GRID_BORDER + grid_row * GRID_SPACING)

def get_grid_coords(display_coords):
	x, y = display_coords
	in_x = x >= GRID_BORDER - GRID_SPACING / 2 and x <= GRID_BORDER + GRID_WIDTH + GRID_SPACING / 2
	in_y = y >= GRID_BORDER - GRID_SPACING / 2 and y <= GRID_BORDER + GRID_HEIGHT + GRID_SPACING / 2

	if not(in_x and in_y):
		return None

	col_idx = int((x + GRID_SPACING / 2 - GRID_BORDER) / GRID_SPACING)
	row_idx = int((y + GRID_SPACING / 2 - GRID_BORDER) / GRID_SPACING)

	return row_idx, col_idx

# draw the game board
BLACK = (0, 0, 0)
BROWN = (165, 42, 42)
WHITE = (255, 255, 255)

def draw_board():
	display_surface.fill(BROWN)

	for row_idx in range(GRID_NUM_ROWS):
		pygame.draw.line(
			display_surface, 
			BLACK, 
			get_display_coords((row_idx, 0)),
			get_display_coords((row_idx, GRID_NUM_COLUMNS - 1)),
			LINE_WIDTH,
		)

	for col_idx in range(GRID_NUM_COLUMNS):
		pygame.draw.line(
			display_surface, 
			BLACK,
			get_display_coords((0, col_idx)),
			get_display_coords((GRID_NUM_ROWS - 1, col_idx)),
			LINE_WIDTH,
		)

	for row_idx, row in enumerate(board.board):
		for col_idx, point in enumerate(row):
			if point.color == Color.UNOCCUPIED:
				continue

			x, y = get_display_coords((row_idx, col_idx))
			pygame.draw.ellipse(
				display_surface,
				BLACK if point.color == Color.BLACK else WHITE,
				(x - STONE_RADIUS, y - STONE_RADIUS, STONE_RADIUS * 2, STONE_RADIUS * 2),
			)

# game loop
while True:
	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
			grid_coords = get_grid_coords(event.pos)
			if grid_coords != None:
				board.place_stone(grid_coords)

		elif event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

	draw_board()
	pygame.display.update()