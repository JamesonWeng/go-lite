import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
				logits = tf.reshape(
					conv_output,
					shape=(-1, num_rows, num_cols),
					name='squeeze',
				)

				# we have to compute our own softmax 
				# so that we can ignore illegal moves

				# for stability, we must first subtract the maximum of the logits in each batch element
				# so we can ignore illegal moves
				logits_max = tf.reduce_max(logits, axis=(1, 2), keep_dims=True)
				logger.debug("logits_max shape: {}".format(logits_max.get_shape()))

				scaled_logits = logits - logits_max
				logger.debug("scaled_logits shape: {}".format(scaled_logits))

				exp_logits = tf.exp(scaled_logits)
				logger.debug("exp_logits shape: {}".format(exp_logits.get_shape()))
				
				masked_logits = exponentiated_logits * self.reasonable_moves
				logger.debug("masked_logits shape: {}".format(masked_logits.get_shape()))
				
				sum_logits = tf.reduce_sum(masked_logits, axis=(1, 2), keep_dims=True)
				logger.debug("sum_logits shape: {}".format(sum_logits.get_shape()))

				policy_output = masked_logits / (sum_logits + 1e-7) # add epsilon so don't divide by 0
				logger.debug("policy_output shape: {}".format(policy_output.get_shape()))

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
		self.actions = tf.placeholder(tf.int32, shape=(None, 2), name='actions')
		# advantages = reward + gamma * value(next_state) - value(current_state)
		self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
		# discounted rewards
		self.rewards = tf.placeholder(tf.float32, shape=(None,), name='rewards')
		
		# loss
		with tf.variable_scope('loss'):
			# policy loss
			batch_indices = tf.range(tf.shape(self.actions)[0])[:, tf.newaxis]
			logger.debug("batch_indices shape: {}".format(batch_indices.get_shape()))

			self.action_indices = tf.concat([batch_indices, self.actions], axis=-1)
			logger.debug("indices shape: {}".format(self.action_indices.get_shape()))

			self.action_probabilities = tf.gather_nd(self.policy_output, self.action_indices)
			logger.debug("action_probabilities shape: {}".format(self.action_probabilities.get_shape()))

			policy_loss = -tf.reduce_mean(tf.log(self.action_probabilities) * self.advantages)

			# value loss
			value_loss = tf.reduce_mean((self.rewards - self.value_output) ** 2)

			# reg loss
			reg_loss = tf.losses.get_regularization_loss()
			
			# total loss
			self.loss = policy_loss + value_loss + reg_loss

		self.init_op = tf.global_variables_initializer()

# test
if __name__ == '__main__':
	model = ActorCritic((5, 10))

	with tf.Session() as session:
		session.run(model.init_op)

		actions = [(2, 8), (0, 1)]
		batch_shape = (2, 5, 10, 3)
		batch_input = np.reshape(np.arange(np.prod(batch_shape)), batch_shape)
		is_training = False
		reasonable_moves = np.ones((2, 5, 10))
		reasonable_moves[1, 3, 4] = 0

		policy_output, action_indices, action_probabilities = session.run(
			[model.policy_output, model.action_indices, model.action_probabilities],
			feed_dict={
				model.input: batch_input,
				model.is_training: is_training,
				model.actions: actions,
				model.reasonable_moves: reasonable_moves,
			}
		)

		print("policy_output: {}".format(policy_output))
		print("action_indices: {}".format(action_indices))
		print("action_probabilities: {}".format(action_probabilities))
