from go_implementation import Color, GoGame
from networks import create_graph

from collections import deque
import logging
import numpy as np
import pickle
import tensorflow as tf

# logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# constants & globals
BATCH_SIZE = 16
BOARD_SIZE = (5, 5)
BUFFER_SIZE = int(1e5)
GAMMA = 0.99

replay_buffer = deque(maxlen=BUFFER_SIZE)

graph_info = create_graph(BOARD_SIZE)
graph = graph_info['graph']

input_tensor = graph.get_tensor_by_name(graph_info['input'])
policy_output_node = graph.get_tensor_by_name(graph_info['policy_output'])
value_output_node = graph.get_tensor_by_name(graph_info['value_output'])

is_training = graph.get_tensor_by_name(graph_info['is_training'])
init_op = graph.get_operation_by_name(graph_info['init_op'])
# summary_op = graph.get_operation_by_name(graph_info['summary_op'])


def generate_experiences(session, epsilon=0.05):
	"""
		epsilon-greedy experience replay
	"""
	go_game = GoGame(BOARD_SIZE)

	while not go_game.is_finished():
		board = go_game.board
		current_color = go_game.next_color
		opposite_color = Color.get_opposite_color(current_color)
		input_board = np.concatenate(
			[
				np.expand_dims(board == current_color, -1), 
				np.expand_dims(board == opposite_color, -1), 
				np.expand_dims(board == Color.UNOCCUPIED, -1),
			], 
			axis=-1,
		)
		policy_output, value_output = session.run(
			[policy_output_node, value_output_node],
			feed_dict={input_tensor: [input_board], is_training: False},
		)

		reasonable_moves = go_game.get_reasonable_moves()
		if not np.any(reasonable_moves):
			logger.debug(
				"No reasonable moves for color: {} on turn: {}".format(current_color, go_game.num_turns)
			)
			go_game.pass_turn()
			continue

		move_predictions = policy_output[0] * reasonable_moves
		next_move = np.unravel_index(np.argmax(move_predictions), move_predictions.shape)
		go_game.place_stone(next_move)

	print("Game finished")
	print("Scores: {}".format(go_game.get_scores()))
	print("Winner: {}".format(go_game.get_winner()))
	winner = go_game.get_winner()

	with open('history.pkl', 'wb') as f:
		pickle.dump(go_game.history, f)


with tf.Session(graph=graph) as session:
	session.run(init_op)

	generate_experiences(session)

	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)