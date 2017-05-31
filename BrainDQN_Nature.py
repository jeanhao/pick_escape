# encoding: utf-8

import tensorflow as tf
import numpy as np
import random
from collections import deque
from config import IMG_WIDTH, IMG_HEIGHT
import os
import shutil

# Hyper Parameters:
FRAME_PER_ACTION = 1  # 每多少帧进行一次行动
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 200.  # timesteps to observe before training
EXPLORE = 50000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.0001  # final value of epsilon
INITIAL_EPSILON = 0  # 0.005  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

BACKUP_INTERVAL = 40000

def copyFiles(sourceDir, targetDir):
	for files in os.listdir(sourceDir):
		sourceFile = os.path.join(sourceDir, files)  # 把目录名和文件名链接起来
		targetFile = os.path.join(targetDir, files)
		if os.path.isfile(sourceFile) and sourceFile.find('network-dqn-') > 0:
			shutil.copy(sourceFile, targetFile)

class BrainDQN:

	def __init__(self, actions, game=None):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		self.game = game
		# init Q network
# 		self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()
#
# 		# init Target Q Network
# 		self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()
#
# 		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1), self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2), self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3), self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1), self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]
		self.stateInput, self.QValue, self.W_conv1, \
		self.b_conv1, self.W_conv2, self.b_conv2, \
		self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, \
		self.W_conv2T, self.b_conv2T, self.W_fc1T, self.b_fc1T, self.W_fc2T, \
		 self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = \
		[self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1), \
		 self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2), \
		 self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1), \
		  self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

		self.networks_directory = "saved_networks"
		self.all_networks_directory = "all_networks"
		if not os.path.exists(self.networks_directory):
			os.makedirs(self.networks_directory)
		if not os.path.exists(self.all_networks_directory):
			os.makedirs(self.all_networks_directory)

		checkpoint = tf.train.get_checkpoint_state(self.networks_directory)

		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print("Could not find old network weights")

		self.checkpoint_record = 0

	def createQNetwork(self):
		# network weights
		W_conv1 = self.weight_variable([1, 5, 4, 16])
		b_conv1 = self.bias_variable([16])

		W_conv2 = self.weight_variable([1, 4, 16, 32])
		b_conv2 = self.bias_variable([32])

		W_fc1 = self.weight_variable([288, 512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512, self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# input layer

		stateInput = tf.placeholder("float", [None, IMG_WIDTH, IMG_HEIGHT, 4])

		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, (1, 5)) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, (1, 2)) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		h_pool2_flat = tf.reshape(h_pool2, [-1, 288])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Q Value layer
		QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

		return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float", [None, self.actions])
		self.yInput = tf.placeholder("float", [None])
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), axis=1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


	def trainQNetwork(self):

		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0, BATCH_SIZE):
			if reward_batch[i] == -1:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))  # 每个batch为两个值

		self.trainStep.run(feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})

		# save network every 10000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

		if self.timeStep - self.checkpoint_record >= BACKUP_INTERVAL:  # 每隔一定间隔备份网络
			self.timeStep += BACKUP_INTERVAL
			copyFiles(self.networks_directory, self.all_networks_directory)
			# 删除checkpoint文件
# 			os.remove(self.networks_directory + "/checkpoint")

# 		if self.timeStep % UPDATE_TIME == 0:
# 			self.copyTargetQNetwork()


	def setPerception(self, nextObservation, action, reward):
		# newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
		self.replayMemory.append((self.currentState, action, reward, newState))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
				# Train the network
			if not self.game or self.game.train:
				self.trainQNetwork()

		# print info for debug
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print("TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon)

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		QValue = self.QValue.eval(feed_dict={self.stateInput:[self.currentState]})[0]  # 全连接处理
		action = np.zeros(self.actions)
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1  # do nothing

		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		return action

	def setInitState(self, observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis=2)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W, strides):
		return tf.nn.conv2d(x, W, strides=[1, strides[0], strides[1], 1], padding="SAME")

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

