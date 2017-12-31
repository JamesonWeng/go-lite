from game_display import GoDisplay
from go_implementation import GoGame

import logging
import pygame
import sys
import time

# logging
logging.basicConfig(level=logging.DEBUG)

# set up the window
pygame.init()
display_surface = pygame.display.set_mode((1000, 500))
pygame.display.set_caption("Go Lite")

go_game = GoGame(board_size=(5, 10))
go_display = GoDisplay(
	go_game=go_game,
	display_coords=(50, 50),
	display_size=(200, 500),
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

	if go_game.is_finished():
		logger.debug(go_game.get_scores())
		break