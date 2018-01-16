from game_display import DisplayManager, InteractiveBoardDisplay, PassButton
from go_implementation import Color, GoGame

import logging
import pygame
import sys
import time

# logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SPACING = 50

# set up the windows & display elements
pygame.init()

# go game 
go_game = GoGame(board_size=(5, 10))

# display elements
board_display = InteractiveBoardDisplay(
	go_game=go_game,
	display_coords=(SPACING, SPACING),
	display_size=(200, 500),
)
pass_button = PassButton(
	go_game=go_game,
	display_coords=(board_display.x2 + 50, board_display.y1),
	display_size=(50, 100),
)

# display manager
display_manager = DisplayManager(
	display_objects=[board_display, pass_button],
	display_size=(pass_button.x2 + SPACING, board_display.y2 + SPACING),
	stop_condition=lambda: go_game.is_finished(),
)

# run game
display_manager.main_loop()

# print results
print("Game finished! Scoring below:")
print("Scores: {}".format(go_game.get_scores()))
print("Winner: {}".format(go_game.get_winner()))