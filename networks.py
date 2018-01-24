import logging
import math
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
		# hyperparams
		num_rows, num_cols = board_size
		gradient_clip = 50.
		learning_rate = 1e-4
		reg_const = 1e-4

		# forward pass placeholders
		input_tensor = tf.placeholder(
			tf.float32, 
			shape=(None, num_rows, num_cols, 3),
			name='input',
		)
		reasonable_moves = tf.placeholder(
			tf.bool,
			shape=(None, num_rows, num_cols),
			name='reasonable_moves',
		)
		is_training = tf.placeholder(tf.bool, name='is_training')

		# L2 regularization
		regularizer = tf.contrib.layers.l2_regularizer(reg_const)

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
						training=is_training,
						name='batch_norm{}'.format(i),
					)
				if activation:
					curr = tf.nn.relu(curr, name='relu{}'.format(i))
			return curr

		with tf.variable_scope('block1'):
			curr = create_conv_block(
				inputs=input_tensor, 
				filters=32, 
				layers=3, 
			)

		with tf.variable_scope('block2'):
			curr = create_conv_block(
				inputs=curr, 
				filters=64, 
				layers=3, 
			)

		with tf.variable_scope('block3'):
			shared_network_output = create_conv_block(
				inputs=curr,
				filters=128, 
				layers=3, 
			)

		# policy network output
		with tf.variable_scope('policy_subnetwork'):
			# no ReLU/BatchNorm before softmax
			curr = create_conv_block(
				inputs=shared_network_output,
				filters=1,
				layers=1,
				activation=False,
				batch_norm=False,
			)
			# remove last dimension of size 1
			logits = tf.reshape(
				curr,
				shape=(-1, num_rows, num_cols),
				name='squeeze',
			)

			# we have to compute our own softmax so that we can ignore illegal/bad moves
			# for stability, we must first subtract the maximum of the logits in each batch element

			# we mask the logits so that we only take max of the values of reasonable moves
			zeros = tf.zeros(tf.shape(reasonable_moves), dtype=tf.float32)
			infinity_mask = tf.where(
				reasonable_moves,
				zeros,
				math.inf + zeros, # addition is for broadcasting
			)
			logger.debug("infinity_mask shape: {}".format(infinity_mask.get_shape()))

			masked_logits = logits - infinity_mask
			logger.debug("masked_logits shape: {}".format(masked_logits.get_shape()))

			max_logits = tf.reduce_max(masked_logits, axis=(1, 2), keep_dims=True)
			logger.debug("max_logits shape: {}".format(max_logits.get_shape()))

			scaled_logits = masked_logits - max_logits
			logger.debug("scaled_logits shape: {}".format(scaled_logits))

			exp_logits = tf.exp(scaled_logits)
			logger.debug("exp_logits shape: {}".format(exp_logits.get_shape()))
			
			# we mask again to actually zero out the moves we don't consider
			masked_exp_logits = exp_logits * tf.to_float(reasonable_moves)
			logger.debug("masked_exp_logits shape: {}".format(masked_exp_logits.get_shape()))
			
			sum_exp_logits = tf.reduce_sum(masked_exp_logits, axis=(1, 2), keep_dims=True)
			logger.debug("sum_exp_logits shape: {}".format(sum_exp_logits.get_shape()))

			policy_output = masked_exp_logits / (sum_exp_logits + 1e-7) # add epsilon so don't divide by 0
			logger.debug("policy_output shape: {}".format(policy_output.get_shape()))

		# value network output
		with tf.variable_scope('value_subnetwork'):
			# no ReLU/BatchNorm before final output
			curr = create_conv_block(
				inputs=shared_network_output,
				filters=1,
				layers=1,
				activation=False,
				batch_norm=False,
			)
			curr = tf.layers.average_pooling2d(
				inputs=curr,
				strides=1,
				pool_size=curr.get_shape()[1:3],
			)
			value_output = tf.reshape(curr, shape=(-1,))

		# LOSS CALCULATION

		# pass in the chosen action for this state: tuple of (row_idx, col_idx)
		actions = tf.placeholder(tf.int32, shape=(None, 2), name='actions')
		# we pass in the advantages as well as rewards
		# because we don't want to compute gradient for the value network
		# when we use advantages in the policy loss
		advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
		# discounted rewards
		rewards = tf.placeholder(tf.float32, shape=(None,), name='rewards')
		
		# loss
		with tf.variable_scope('loss'):
			# policy loss
			batch_indices = tf.range(tf.shape(actions)[0])[:, tf.newaxis]
			logger.debug("batch_indices shape: {}".format(batch_indices.get_shape()))

			action_indices = tf.concat([batch_indices, actions], axis=-1)
			logger.debug("indices shape: {}".format(action_indices.get_shape()))

			# we take the log of the softmax in a more numerically stable way
			# sum_logits is guaranteed to be at least 1 
			# since we scaled our logits to have max = 0
			log_softmax = scaled_logits - tf.log(sum_exp_logits)
			logger.debug("log_softmax shape: {}".format(log_softmax.get_shape()))

			log_action_probs = tf.gather_nd(log_softmax, action_indices)
			logger.debug("log_action_probs shape: {}".format(log_action_probs.get_shape()))

			policy_loss = -tf.reduce_mean(log_action_probs * advantages)

			# value loss
			value_loss = tf.reduce_mean((rewards - value_output) ** 2)

			# reg loss
			reg_loss = tf.losses.get_regularization_loss()
			
			# total loss
			loss = policy_loss + value_loss + reg_loss

			# backprop
			optimizer = tf.train.AdamOptimizer(learning_rate)
			# optimizer = tf.train.MomentumOptimizer(
			# 	learning_rate=learning_rate,
			# 	momentum=0.6,
			# 	use_nesterov=True,
			# )
			grads_and_vars = optimizer.compute_gradients(loss)
			optimize = optimizer.apply_gradients(grads_and_vars)

		# keep pointers to relevant nodes
		self.input = input_tensor
		self.reasonable_moves = reasonable_moves
		self.is_training = is_training

		self.logits = logits
		self.infinity_mask = infinity_mask
		self.masked_logits = masked_logits
		self.max_logits = max_logits
		self.scaled_logits = scaled_logits
		self.exp_logits = exp_logits
		self.masked_exp_logits = masked_exp_logits
		self.sum_exp_logits = sum_exp_logits
		self.policy_output = policy_output
		self.value_output = value_output

		self.actions = actions
		self.advantages = advantages
		self.rewards = rewards

		self.action_indices = action_indices
		self.log_softmax = log_softmax
		self.log_action_probs = log_action_probs
		self.policy_loss = policy_loss
		self.value_loss = value_loss
		self.reg_loss = reg_loss
		self.loss = loss
		self.optimize = optimize

		self.init_op = tf.global_variables_initializer()


# test
if __name__ == '__main__':
	logger.setLevel(logging.DEBUG)
	model = ActorCritic((5, 10))

	with tf.Session() as session:
		session.run(model.init_op)

		actions = [(2, 8), (0, 1)]
		batch_shape = (2, 5, 10, 3)
		batch_input = np.reshape(np.arange(np.prod(batch_shape)), batch_shape)
		is_training = False
		reasonable_moves = np.ones((2, 5, 10))
		reasonable_moves[1, 3, 4] = 0

		advantages = [2, 5]
		rewards = [0.99 ** (batch_shape[0] - i) for i in range(batch_shape[0])]

		tensors = np.array([
			["infinity_mask", model.infinity_mask],
			["masked_logits", model.masked_logits],
			["max_logits", model.max_logits],
			["scaled_logits", model.scaled_logits],
			["exp_logits", model.exp_logits],
			["masked_exp_logits", model.masked_exp_logits],
			["sum_exp_logits", model.sum_exp_logits],
			["policy_output", model.policy_output],
			["value_output", model.value_output],
			["action_indices", model.action_indices],
			["log_softmax", model.log_softmax],
			["log_action_probs", model.log_action_probs],
			["policy_loss", model.policy_loss],
			["value_loss", model.value_loss],
			["reg_loss", model.reg_loss],
			["loss", model.loss],
			["optimize", model.optimize],
		])

		tensor_values = session.run(
			tensors[:, 1].tolist(),
			feed_dict={
				model.input: batch_input,
				model.is_training: is_training,
				model.actions: actions,
				model.reasonable_moves: reasonable_moves,
				model.advantages: advantages,
				model.rewards: rewards,
			}
		)

		for (tensor_name, value) in zip(tensors[:, 0], tensor_values):
			print("tensor: {}".format(tensor_name))
			print("{}\n".format(value))

		summary_writer = tf.summary.FileWriter('logs', graph=session.graph)