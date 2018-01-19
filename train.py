from go_implementation import Color, GoGame
from networks import ActorCritic

from collections import deque
import datetime
import logging
import numpy as np
import os
import pickle
import sys
import random
import tensorflow as tf

# logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# constants & globals
BATCH_SIZE = 8
BUFFER_SIZE = int(1e5)
BOARD_SIZE = (5, 5)
GAMMA = 0.99

CHECKPOINT_DIR = './checkpoints/{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)

class ExperienceBuffer(object):
	def __init__(self):
		self._buffers = {
			'actions': [],
			'advantages': [],
			'boards': [],
			'reasonable_moves': [],
			'rewards': [],
		}

	def append(self, actions, advantages, boards, reasonable_moves, rewards):
		self._buffers['actions'] += list(actions)
		self._buffers['advantages'] += list(advantages)
		self._buffers['boards'] += list(boards)
		self._buffers['reasonable_moves'] += list(reasonable_moves)
		self._buffers['rewards'] += list(rewards)

		# assert same length
		assert(len(set(map(len, self._buffers.values()))))

		# chop down to max length
		curr_length = self.get_length()
		if curr_length > BUFFER_SIZE:
			to_chop = curr_length - BUFFER_SIZE
			for k, v in self._buffers.items():
				self._buffers[k] = v[to_chop:]

	def get_batch(self):
		start_idx = random.randint(0, len(self._buffers['actions']) - BATCH_SIZE)
		end_idx = start_idx + BATCH_SIZE

		actions = self._buffers['actions'][start_idx: end_idx]
		advantages = self._buffers['advantages'][start_idx: end_idx]
		boards = self._buffers['boards'][start_idx: end_idx]
		reasonable_moves = self._buffers['reasonable_moves'][start_idx: end_idx]
		rewards = self._buffers['rewards'][start_idx: end_idx]
		return actions, advantages, boards, reasonable_moves, rewards

	def get_length(self):
		return len(self._buffers['actions'])


experience_buffer = ExperienceBuffer()


def preprocess_board(board, current_color):
	"""
		preprocess go board for input to the actor-critic model
	"""
	def get_mask(board, color):
		return np.expand_dims(board == color, -1).astype(float)

	opposite_color = Color.get_opposite_color(current_color)
	to_concatenate = [
		get_mask(board, current_color),
		get_mask(board, opposite_color),
		get_mask(board, Color.UNOCCUPIED),
	]
	processed_board = np.concatenate(to_concatenate, axis=-1)
	processed_board -= 0.5
	return processed_board

def generate_experiences(session, model):
	"""
		generate experiences with experience replay
	"""
	go_game = GoGame(BOARD_SIZE)

	history = {}
	for color in [Color.BLACK, Color.WHITE]:
		history[color] = {
			'actions': [],
			'boards': [],
			'reasonable_moves': [],
			'values': [],
		}

	# simulate game
	while not go_game.is_finished():
		# compute inputs
		current_color = go_game.next_color
		board = preprocess_board(go_game.board, current_color)
		reasonable_moves = go_game.get_reasonable_moves()

		# forward pass
		policy_output, value_output = session.run(
			[model.policy_output, model.value_output],
			feed_dict={
				model.input: [board], 
				model.is_training: False,
				model.reasonable_moves: [reasonable_moves],
			}
		)

		# since batch size 1, index the 0th element
		policy_output = policy_output[0]
		value_output = value_output[0]

		# if no reasonable moves, we "pass"
		if not np.any(reasonable_moves):
			logger.debug("No reasonable moves for color: {} on turn: {}".format(
				current_color, go_game.num_turns)
			)
			go_game.pass_turn()
			continue

		# otherwise, sample a move by the probability predicted by the model
		linear_idx = np.random.choice(
			policy_output.size, 
			p=policy_output.ravel(),
		)
		next_move = np.unravel_index(linear_idx, policy_output.shape)
		
		valid_move = go_game.place_stone(next_move)
		if not valid_move:
			logger.error(
				"tried to play {} for color {} on board {} with predictions {}"
				.format(next_move, current_color, go_game.board, policy_output)
			)
			return

		# save some values
		player_history = history[current_color]
		player_history['actions'].append(next_move)
		player_history['boards'].append(board)
		player_history['reasonable_moves'].append(reasonable_moves)
		player_history['values'].append(value_output)

	# find winner
	winner = go_game.get_winner()
	logger.info("Game finished in {} turns, winner is: {}".format(go_game.num_turns, winner))

	# calculate advantages and rewards
	for color in [Color.BLACK, Color.WHITE]:
		final_reward = 1 if color == winner else -1
		player_history = history[color]
		num_turns = len(player_history['actions'])

		# discounted rewards
		rewards = np.array([
			final_reward * GAMMA ** (num_turns - idx)
			for idx in range(num_turns)
		])

		# advantages
		advantages = rewards - np.array(player_history['values'])

		# add to buffer
		logger.debug("saving experience for {}".format(color))
		logger.debug("advantages shape: {}".format(advantages.shape))
		logger.debug("advantages: {}".format(advantages[:5]))
		logger.debug("rewards shape: {}".format(rewards.shape))
		logger.debug("rewards: {}".format(rewards[:5]))

		experience_buffer.append(
			actions=player_history['actions'],
			advantages=advantages,
			boards=player_history['boards'],
			reasonable_moves=player_history['reasonable_moves'],
			rewards=rewards,
		)
	return go_game

# create the model
graph = tf.Graph()
with graph.as_default():
	model = ActorCritic(BOARD_SIZE)
	saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
	NUM_EXPERIENCES = 1

	checkpoint_interval = 10
	global_step = 0
	summary_writer = tf.summary.FileWriter('logs', graph=session.graph)

	session.run(model.init_op)

	while True:
		for _ in range(NUM_EXPERIENCES):
			go_game = generate_experiences(session, model)
			logger.debug("experience buffer length: {}".format(experience_buffer.get_length()))

		for _ in range(experience_buffer.get_length() // BATCH_SIZE):
			global_step += 1

			actions, advantages, boards, reasonable_moves, rewards = experience_buffer.get_batch()
			loss, _ = session.run(

				[model.loss, model.optimize],
				feed_dict = {
					model.input: boards,
					model.is_training: True,
					model.actions: actions,
					model.advantages: advantages,
					model.reasonable_moves: reasonable_moves,
					model.rewards: rewards,
				},
			)
			logger.info("global_step: {} loss: {}".format(global_step, loss))

			if global_step % checkpoint_interval == 0:
				logger.debug("saving model")
				saver.save(session, os.path.join(CHECKPOINT_DIR, 'model'), global_step=global_step)

				with open('history.pkl', 'wb') as f:
					pickle.dump(go_game.history, f)
