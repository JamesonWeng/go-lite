import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)

def create_conv_block(inputs, filters, layers):
	current_tensor = inputs
	for i in range(layers):
		current_tensor = tf.layers.conv2d(
			inputs=current_tensor,
			filters=filters,
			kernel_size=(3, 3),
			padding='same',
			activation=tf.nn.relu,
			name='conv{}'.format(i),
		)
	return current_tensor


batch_size = 16
num_rows, num_cols = (5, 10)
num_channels = 3
input_size=(batch_size, num_rows, num_cols, num_channels)

graph = tf.Graph()
with graph.as_default():
	input_tensor = tf.placeholder(
		tf.float32, 
		shape=(batch_size, num_rows, num_cols, 3),
		name='input',
	)

	with tf.variable_scope('block1'):
		block1_output = create_conv_block(input_tensor, 32, 3)

	with tf.variable_scope('block2'):
		block2_output = create_conv_block(block1_output, 64, 3)

	with tf.variable_scope('block3'):
		block3_output = create_conv_block(block2_output, 128, 3)

	# the output is a probability distribution across the board
	# with two channels: one for black's best move, one for white's best move
	with tf.variable_scope('block4'):
		conv_output = create_conv_block(block3_output, 2, 1)
		logging.debug("conv_output shape: {}".format(conv_output.get_shape()))

		# collapse height and width to perform softmax across both
		reshape_output = tf.reshape(
			conv_output,
			shape=(batch_size, num_rows * num_cols, 2),
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
			shape=(batch_size, num_rows, num_cols, 2),
			name='restore_shape',
		)
		logging.debug("restore_shape shape: {}".format(reshape_output.get_shape()))

		output_tensor = reshape_output

	init_op = tf.global_variables_initializer()
	merged_summary_op = tf.summary.merge_all()


with tf.Session(graph=graph) as session:
	session.run(init_op)
	res = session.run(output_tensor, feed_dict={input_tensor: np.zeros(input_size)})
	# print(res)
	# print(res.shape)
	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)