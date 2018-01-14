import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)

def create_conv_block(inputs, filters, layers, is_training, activation=True, batch_norm=True):
	curr = inputs
	for i in range(layers):
		curr = tf.layers.conv2d(
			inputs=curr,
			filters=filters,
			kernel_size=(3, 3),
			padding='same',
			use_bias=False,
			activation=None,
			name='conv{}'.format(i),
		)
		if batch_norm:
			curr = tf.layers.batch_normalization(
				inputs=curr,
				training=is_training,
				name='batch_norm{}'.format(i),
			)
		if activation:
			curr = tf.nn.relu(curr, name='relu{}'.format(i))
	return curr

def create_graph(board_size):
	"""
		Create the graph for the policy network.

		The output is a probability distribution across the board
		with two channels: one for black's best move, one for white's best move.

		For now, model doesn't pass its turns.
	"""

	logging.debug('Defining the graph:')

	num_rows, num_cols = board_size
	input_size=(None, num_rows, num_cols, 3)

	graph = tf.Graph()

	with graph.as_default():
		input_tensor = tf.placeholder(
			tf.float32, 
			shape=input_size,
			name='input',
		)

		is_training = tf.placeholder(tf.bool, name='is_training')

		with tf.variable_scope('block1'):
			block1_output = create_conv_block(
				inputs=input_tensor, 
				filters=32, 
				layers=3, 
				is_training=is_training)

		with tf.variable_scope('block2'):
			block2_output = create_conv_block(
				inputs=block1_output, 
				filters=64, 
				layers=3, 
				is_training=is_training,
			)

		with tf.variable_scope('block3'):
			block3_output = create_conv_block(
				inputs=block2_output, 
				filters=128, 
				layers=3, 
				is_training=is_training,
			)

		# output block
		# we don't want any ReLU or BatchNorm before the output as this will mess up softmax
		with tf.variable_scope('block4'):
			conv_output = create_conv_block(
				inputs=block3_output, 
				filters=2, 
				layers=1, 
				is_training=is_training,
				activation=False,
				batch_norm=False,
			)
			logging.debug("conv_output shape: {}".format(conv_output.get_shape()))

			# collapse height and width to perform softmax across both
			reshape_output = tf.reshape(
				conv_output,
				shape=(-1, num_rows * num_cols, 2),
				name='collapse_shape',
			)
			logging.debug("collapse_shape shape: {}".format(reshape_output.get_shape()))

			# apply softmax
			softmax_output = tf.nn.softmax(
				reshape_output,
				dim=1,
				name='softmax',
			)
			logging.debug("softmax_output shape: {}".format(softmax_output.get_shape()))

			# undo reshape
			reshape_output = tf.reshape(
				softmax_output,
				shape=(-1, num_rows, num_cols, 2),
				name='restore_shape',
			)
			logging.debug("restore_shape shape: {}".format(reshape_output.get_shape()))

			output_tensor = reshape_output

		init_op = tf.global_variables_initializer()
		# summary_op = tf.summary.merge_all()

	return {
		'graph': graph,
		'input_node': input_tensor.name,
		'output_node': output_tensor.name,
		'is_training': is_training.name,
		'init_op': init_op.name,
		# 'summary_op': summary.name,
	}


graph_info = create_graph(board_size=(5, 10))
graph = graph_info['graph']
input_tensor = graph.get_tensor_by_name(graph_info['input_node'])
output_tensor = graph.get_tensor_by_name(graph_info['output_node'])
is_training = graph.get_tensor_by_name(graph_info['is_training'])
init_op = graph.get_operation_by_name(graph_info['init_op'])
# summary_op = graph.get_operation_by_name(graph_info['summary_op'])

with tf.Session(graph=graph) as session:
	session.run(init_op)

	res = session.run(
		output_tensor, 
		feed_dict={input_tensor: np.zeros((16, 5, 10, 3)), is_training: False},
	)

	# print(res)
	# print(res.shape)
	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)