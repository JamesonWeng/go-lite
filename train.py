from go_implementation import Color, GoGame
from networks import create_graph

from collections import deque
import numpy as np
import tensorflow as tf

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

def generate_experiences(session):
	go_game = GoGame(BOARD_SIZE)

	while not go_game.is_finished():
		board = go_game.board
		input_board = np.concatenate(
			[
				np.expand_dims(board == Color.BLACK, -1), 
				np.expand_dims(board == Color.WHITE, -1), 
				np.expand_dims(board == Color.UNOCCUPIED, -1),
			], 
			axis=-1,
		)
		policy_output, value_output = session.run(
			[policy_output_node, value_output_node],
			feed_dict={input_tensor: [input_board], is_training: False},
		)

		output_channel = 0 if go_game.next_color == Color.BLACK else 1
		reasonable_moves = go_game.get_reasonable_moves()

		move_predictions = policy_output[0, :, :, output_channel] * reasonable_moves

		break


with tf.Session(graph=graph) as session:
	session.run(init_op)

	generate_experiences(session)

	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)