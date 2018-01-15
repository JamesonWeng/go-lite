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
			use_bias=False if batch_norm else True,
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

		The input is a 4D tensor with 3 channels representing:
		- black pieces
		- white pieces
		- empty

		The policy output is a probability distribution across the board
		with two channels: one for black's best move, one for white's best move.

		The value output is a single number representing the expected future discounted rewards.

		For now, model doesn't pass its turns.
	"""

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

		# policy network output
		with tf.variable_scope('policy_subnetwork'):
			# no ReLU/BatchNorm before softmax
			conv_output = create_conv_block(
				inputs=block3_output, 
				filters=2, 
				layers=1, 
				is_training=is_training,
				activation=False,
				batch_norm=False,
			)
			# collapse height and width to perform softmax across both
			reshape_output = tf.reshape(
				conv_output,
				shape=(-1, num_rows * num_cols, 2),
				name='collapse_shape',
			)
			# apply softmax
			softmax_output = tf.nn.softmax(
				reshape_output,
				dim=1,
				name='softmax',
			)
			# undo reshape
			reshape_output = tf.reshape(
				softmax_output,
				shape=(-1, num_rows, num_cols, 2),
				name='restore_shape',
			)
			policy_output = reshape_output

		# value network output
		with tf.variable_scope('value_subnetwork'):
			# no ReLU/BatchNorm before final output
			conv_output = create_conv_block(
				inputs=block3_output,
				filters=1,
				layers=1,
				is_training=is_training,
				activation=False,
				batch_norm=False,
			)
			pool_output = tf.layers.average_pooling2d(
				inputs=conv_output,
				strides=1,
				pool_size=conv_output.get_shape()[1:3],
			)
			flatten_output = tf.layers.flatten(inputs=pool_output)
			value_output = flatten_output
		
		init_op = tf.global_variables_initializer()
		# summary_op = tf.summary.merge_all()

	return {
		'graph': graph,
		'input': input_tensor.name,
		'policy_output': policy_output.name,
		'value_output': value_output.name,
		'init_op': init_op.name,
		'is_training': is_training.name,
		# 'summary_op': summary.name,
	}