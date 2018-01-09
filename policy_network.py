import logging
import numpy as np
import tensorflow as tf

def create_conv_block(inputs, filters, layers):
	current_tensor = inputs
	for i in range(layers):
		current_tensor = tf.layers.conv2d(
			inputs=current_tensor,
			filters=32,
			kernel_size=(3, 3),
			padding='same',
			activation=tf.nn.relu,
			name='conv{}'.format(i),
		)
	return current_tensor


batch_size = 16
num_rows, num_cols = (5, 10)
input_size=(batch_size, num_rows, num_cols, 3)

graph = tf.Graph()
with graph.as_default():
	input_tensor = tf.placeholder(
		tf.float32, 
		shape=(batch_size, num_rows, num_cols, 3),
		name='input',
	)

	with tf.variable_scope('block1'):
		output_tensor = create_conv_block(input_tensor, 32, 3)

	with tf.variable_scope('block2'):
		output_tensor = create_conv_block(output_tensor, 64, 3)

	with tf.variable_scope('block3'):
		output_tensor = create_conv_block(output_tensor, 128, 3)

	init_op = tf.global_variables_initializer()
	merged_summary_op = tf.summary.merge_all()


with tf.Session(graph=graph) as session:
	session.run(init_op)
	res = session.run(output_tensor, feed_dict={input_tensor: np.zeros(input_size)})
	print(res)
	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)