from game_display import BURNT_UMBER, Button, GoDisplay, PassButton
from go_implementation import Color, GoGame

import logging
import pygame
import sys
import time

# logging
logging.basicConfig(level=logging.DEBUG)

# set up the windows & display elements
pygame.init()


SPACING = 50

go_game = GoGame(board_size=(5, 10))
go_display = GoDisplay(
	go_game=go_game,
	coords=(SPACING, SPACING),
	size=(200, 500),
)


pass_button = PassButton(
	go_game=go_game,
	coords=(go_display.x2 + 50, go_display.y1),
	size=(50, 100),
)

display_size = (pass_button.x2 + SPACING, go_display.y2 + SPACING)
display_surface = pygame.display.set_mode(display_size)
pygame.display.set_caption("Go Lite")

display_objects = [go_display, pass_button]

# game loop
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

		for display_obj in display_objects:
			display_obj.handle_event(event)

	display_surface.fill(BURNT_UMBER)
	for display_obj in display_objects:
		display_obj.draw(display_surface)
	pygame.display.update()

	if go_game.is_finished():
		logging.debug('Game finished! Scoring below:')
		logging.debug(go_game.get_scores())
		logging.debug('winner: {}'.format(go_game.get_winner()))
		break