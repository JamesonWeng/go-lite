from game_display import DisplayManager, HistoryDisplay

import logging
import pickle
import pygame
import sys

SPACING = 50

logging.basicConfig(level=logging.DEBUG)
pygame.init()

with open(sys.argv[1], 'rb') as f:
	history = pickle.load(f)

history_display = HistoryDisplay(
	board_size=history[0].shape,
	display_coords=(SPACING, SPACING),
	display_size=(500, 500),
	history=history,
)

display_manager = DisplayManager(
	display_objects=[history_display],
	display_size=(history_display.x2 + SPACING, history_display.y2 + SPACING),
)

display_manager.main_loop()