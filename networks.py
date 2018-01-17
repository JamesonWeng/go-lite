import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class ActorCritic(object):
	"""
		Create the policy and value networks, which share input.

		The input is a tensor of shape (batch_size, num_rows, num_cols, 3),
		where the entries in the 3 channels are respectively set to:
		- 1 if occupied by stone of current color
		- 1 if occupied by stone of opposite color
		- 1 if unoccupied

		The policy output is a probability distribution across reasonable moves on the board,
		for the current color's best move.

		The value output is a single number representing the expected future discounted rewards.

		For now, model doesn't pass any its turns.
	"""

	def __init__(self, board_size):
		num_rows, num_cols = board_size
		self.input_shape = (None, num_rows, num_cols, 3)

		# forward pass placeholders
		self.input = tf.placeholder(
			tf.float32, 
			shape=self.input_shape,
			name='input',
		)
		self.reasonable_moves = tf.placeholder(
			tf.float32,
			shape=(None, num_rows, num_cols),
			name='reasonable_moves',
		)
		self.is_training = tf.placeholder(tf.bool, name='is_training')

		# optimizer & regularizer
		optimizer = tf.train.AdamOptimizer(1e-4)
		regularizer = tf.contrib.layers.l2_regularizer(1e-4)

		# ops
		def create_conv_block(inputs, filters, layers, activation=True, batch_norm=True):
			curr = inputs
			for i in range(layers):
				curr = tf.layers.conv2d(
					inputs=curr,
					filters=filters,
					kernel_size=(3, 3),
					kernel_regularizer=regularizer,
					padding='same',
					use_bias=False if batch_norm else True,
					activation=None,
					name='conv{}'.format(i),
				)
				if batch_norm:
					curr = tf.layers.batch_normalization(
						inputs=curr,
						training=self.is_training,
						name='batch_norm{}'.format(i),
					)
				if activation:
					curr = tf.nn.relu(curr, name='relu{}'.format(i))
			return curr

		def forward_pass(input):
			with tf.variable_scope('block1'):
				block1_output = create_conv_block(
					inputs=self.input, 
					filters=32, 
					layers=3, 
				)

			with tf.variable_scope('block2'):
				block2_output = create_conv_block(
					inputs=block1_output, 
					filters=64, 
					layers=3, 
				)

			with tf.variable_scope('block3'):
				block3_output = create_conv_block(
					inputs=block2_output,
					filters=128, 
					layers=3, 
				)

			# policy network output
			with tf.variable_scope('policy_subnetwork'):
				# no ReLU/BatchNorm before softmax
				conv_output = create_conv_block(
					inputs=block3_output,
					filters=1,
					layers=1,
					activation=False,
					batch_norm=False,
				)
				# remove last dimension of size 1
				reshape_output = tf.reshape(
					conv_output,
					shape=(-1, num_rows, num_cols),
					name='flatten',
				)

				# we perform our own softmax
				# so we can ignore illegal moves
				exponentiated_output = tf.exp(reshape_output)
				logger.debug("exponentiated_output shape: {}".format(exponentiated_output.get_shape()))
				
				masked_output = exponentiated_output * self.reasonable_moves
				logger.debug("masked_output shape: {}".format(masked_output.get_shape()))
				
				sum_output = tf.reduce_sum(masked_output, axis=(1, 2))
				logger.debug("sum_output shape: {}".format(sum_output.get_shape()))

				policy_output = masked_output / (sum_output[:, tf.newaxis, tf.newaxis] + 1e-7)
				logger.debug("policy_output shape: {}".format(policy_output.get_shape()))


				# collapse height and width to perform softmax across both
				# reshape_output = tf.reshape(
				# 	conv_output,
				# 	shape=(-1, num_rows * num_cols),
				# 	name='collapse_shape',
				# )
				# # apply softmax
				# softmax_output = tf.nn.softmax(
				# 	reshape_output,
				# 	dim=1,
				# 	name='softmax',
				# )
				# # undo reshape
				# reshape_output = tf.reshape(
				# 	softmax_output,
				# 	shape=(-1, num_rows, num_cols),
				# 	name='restore_shape',
				# )
				# policy_output = reshape_output

			# value network output
			with tf.variable_scope('value_subnetwork'):
				# no ReLU/BatchNorm before final output
				conv_output = create_conv_block(
					inputs=block3_output,
					filters=1,
					layers=1,
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
			return policy_output, value_output

		self.policy_output, self.value_output = forward_pass(self.input)

		# LOSS CALCULATION

		# pass in the chosen action for this state: tuple of (row_idx, col_idx)
		self.action = tf.placeholder(tf.float32, shape=(None, 2), name='action')
		# pass in the board state after the action from policy network
		self.post_action_board = tf.placeholder(tf.float32, shape=self.input_shape)
		# advantages = reward + gamma * value(next_state) - value(current_state)
		self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
		# discounted rewards
		self.rewards = tf.placeholder(tf.float32, shape=(None,), name='rewards')
		
		# loss
		with tf.variable_scope('loss'):
			policy_loss = self.policy_output

			value_loss = tf.reduce_mean(self.advantages ** 2)

			reg_loss = tf.losses.get_regularization_loss()
			loss = value_loss + reg_loss


		self.init_op = tf.global_variables_initializer()