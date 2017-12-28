from go_display import GoDisplay

import logging
import pygame
import sys
import time

# logging
logging.basicConfig(level=logging.DEBUG)

# set up the window
pygame.init()
display_surface = pygame.display.set_mode((1000, 500))
pygame.display.set_caption('Go Lite')

go_display = GoDisplay(
	num_rows=5, num_cols=10, grid_height=200, grid_width=500, grid_x1=50, grid_y1=50,
)

# game loop
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

		go_display.handle_event(event)

	go_display.draw(display_surface)
	pygame.display.update()