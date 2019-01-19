# some code fragments borrowed from Jaromir Janisch, 2017

# from __future__ import print_function

import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import gym, time, random, threading
import gym_skynet

from keras.models import *
from keras.layers import *
from keras import backend as K

import itertools

#-- constants
ENV = 'Skynet-v0'

RUN_TIME = 60
THREADS = 8
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99
N_STEP_RETURN = 1
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.0
EPS_STOP  = 0.0
EPS_STEPS = 20000

MIN_BATCH = 32
LEARNING_RATE = 5e-2

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

MODEL_VERSION = 1	# state-dependent

DEBUG = False
VERBOSE = True

#---------
class Brain:
	train_queue = [ [[], [], []], [], [], [[], [], []], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.train_iteration = 0

		self.session = tf.Session()
		# self.session = K.get_session()
		# self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.topology_shape = OBSERVATION_SPACE.spaces["topology"].shape
		self.routes_shape = OBSERVATION_SPACE.spaces["routes"].shape
		self.reachability_shape = OBSERVATION_SPACE.spaces["reachability"].shape
		self.action_shape_height = int(ACTION_SPACE.high[0])
		self.action_shape_width = int(ACTION_SPACE.high[1])

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

	def _build_model(self):
		topology_input = Input(shape=self.topology_shape)
		routes_input = Input(shape=self.routes_shape)
		reachability_input = Input(shape=self.reachability_shape)

		merged_input = Concatenate(axis=1)([topology_input, routes_input, reachability_input])
		flattened_input = Flatten()(merged_input)
		dense_layer_1 = Dense(256, activation='relu')(flattened_input)
		dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)
		# K.print_tensor(dense_layer_1, message='dense layer weights = ')

		_out_actions = Dense(self.action_shape_height*self.action_shape_width, activation='softmax')(dense_layer_2)
		out_actions = Reshape((self.action_shape_height, self.action_shape_width))(_out_actions)
		out_value = Dense(1, activation='linear')(dense_layer_2)

		model = Model(inputs=[topology_input, routes_input, reachability_input], outputs=[out_actions, out_value])

		# # input layer dimensions: number of flows * number of links
		# l_input = Input(batch_shape=(None, int(STATE[0])*int(STATE[1])))
		# l_dense_1 = Dense(32, activation='relu')(l_input)
		# l_dense_2 = Dense(16, activation='relu')(l_dense_1)

		# # output layer dimensions: number of flows * number of links
		# out_actions = Dense(int(ACTION.high[0])*int(ACTION.high[1]), activation='softmax')(l_dense_2)
		# out_value   = Dense(1, activation='linear')(l_dense_2)

		# model = Model(inputs=[l_input], outputs=[out_actions, out_value])

		model._make_predict_function()	# have to initialize before threading
		
		if DEBUG:
			print(model.summary())

		return model

	def _build_graph(self, model):
		# s_t = tf.placeholder(tf.float32, shape=(None, int(STATE[0])*int(STATE[1])))
		# a_t = tf.placeholder(tf.float32, shape=(None, int(ACTION.high[0])*int(ACTION.high[1])))
		# r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
		# prob, avg_value = model(s_t)

		topo_t = tf.placeholder(tf.float32, shape=(None, self.topology_shape[0], self.topology_shape[1]))
		routes_t = tf.placeholder(tf.float32, shape=(None, self.routes_shape[0], self.routes_shape[1]))
		reach_t = tf.placeholder(tf.float32, shape=(None, self.reachability_shape[0], self.reachability_shape[1]))

		action_t = tf.placeholder(tf.float32, shape=(None, self.action_shape_height, self.action_shape_width))
		reward_t = tf.placeholder(tf.float32, shape=(None, 1))

		prob, avg_reward = model([topo_t, routes_t, reach_t])

		log_prob = tf.log(tf.reduce_sum(prob * action_t, axis=1, keepdims=True) + 1e-10)
		advantage = reward_t - avg_reward

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(prob * tf.log(prob + 1e-10), axis=1, keepdims=True)	# maximize entropy (regularization)

		# op_names = [str(op.name) for op in tf.get_default_graph().get_operations()]

		# print(*(str((type(op), op.name, op)) for op in tf.get_default_graph().get_operations()), sep='\n')
		# print_op = tf.Print(action_t,  tf.get_default_graph().get_operations())
		
		# print_op = tf.Print(action_t,  [reward_t])
		# with tf.control_dependencies([print_op]):
		# 	loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
		
		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		# optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(loss_total)

		return topo_t, routes_t, reach_t, action_t, reward_t, minimize

	# def get_weights(self):
	# 	tvars = tf.trainable_variables()
	# 	print(tvars)
	# 	tvars_vals = self.session.run(tvars)
	# 	for var, val in zip(tvars, tvars_vals):
	# 		print(var.name, val)
  	# 	# return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]

	def optimize(self):
		if len(self.train_queue[0][0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0][0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			state, action, reward, state_, state_mask = self.train_queue
			# s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [[], [], []], [], [], [[], [], []], [] ]

		# print "before vstack, topo shape: %s, route shape: %s" % (state[0][0].shape, state[1][0].shape)
		
		self.train_iteration = self.train_iteration + 1

		state_topo = np.vstack(state[0])
		state_routes = np.vstack(state[1])
		state_reach = np.vstack(state[2])
		action = np.vstack(action)
		reward = np.vstack(reward)
		state_topo_ = np.vstack(state_[0])
		state_routes_ = np.vstack(state_[1])
		state_reach_ = np.vstack(state_[2])
		state_mask = np.vstack(state_mask)
		
		# # print "topo: %s, shape: %s" % (str(state_topo), str(state_topo[0].shape))
		# print "routes: %s, shape: %s" % (str(state_routes), str(state_routes.shape))
		# print "action: %s, shape: %s" % (str(action), str(action.shape))
		# # print "reward: %s, shape: %s" % (str(reward), str(reward.shape))

		if len(state_topo) > 5*MIN_BATCH: 
			print("Optimizer alert! Minimizing batch of %d" % len(state_topo))

		avg_reward = self.predict_avg_reward([state_topo_, state_routes_, state_reach_])
		reward = reward + GAMMA_N * avg_reward * state_mask	# set avg_reward to 0 where state_ is terminal state
		# print "state_mask: %s" % str(state_mask)
		# print "reward value: %s" % str(reward)
		
		topo_t, routes_t, reach_t, action_t, reward_t, minimize = self.graph

		# np.save('prev_train_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
		self.session.run(minimize, feed_dict={topo_t: state_topo, routes_t: state_routes, reach_t: state_reach, action_t: action, reward_t: reward})
		# np.save('after_train_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
		# self.get_weights()

	def train_push(self, state, action, reward, state_):
		if DEBUG:
			print("Training Datum: Routes: %s, Action: %s, Reward: %s" % (str(state["routes"]), str(action), str(reward)))

		with self.lock_queue:
			# print "routes shape: %s" % str(state["routes"].shape)
			self.train_queue[0][0].append([state["topology"]])
			self.train_queue[0][1].append([state["routes"]])
			self.train_queue[0][2].append([state["reachability"]])
			self.train_queue[1].append([action])
			self.train_queue[2].append([reward])

			if state_ is None:
				# print "STATE IS NONE"
				self.train_queue[3][0].append([NULL_STATE["topology"]])
				self.train_queue[3][1].append([NULL_STATE["routes"]])
				self.train_queue[3][2].append([NULL_STATE["reachability"]])
				self.train_queue[4].append([0.])
			else:	
				self.train_queue[3][0].append([state_["topology"]])
				self.train_queue[3][1].append([state_["routes"]])
				self.train_queue[3][2].append([state_["reachability"]])
				self.train_queue[4].append([1.])

	def predict(self, state):
		with self.default_graph.as_default():
			prob, avg_reward = self.model.predict(state)
			return prob, avg_reward

	def predict_prob(self, state):
		with self.default_graph.as_default():
			# print state[0].shape
			# print state[1].shape
			# print state[2].shape
			# np.save('predict_prob_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
			prob, avg_reward = self.model.predict(state)
			return prob

	def predict_avg_reward(self, state):
		with self.default_graph.as_default():
			# np.save('predict_avg_reward_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
			prob, avg_reward = self.model.predict(state)
			return avg_reward

#---------
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps, env):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps
		self.env = env

		self.memory = []	# used for n_step return
		self.R = 0.

		self.action_shape_height = int(self.env.action_space.high[0])
		self.action_shape_width = int(self.env.action_space.high[1])

	def getEpsilon(self):
		frames = FRAMES.next()
		if frames >= self.eps_steps:
			if frames == self.eps_steps:
				if VERBOSE and not TESTING:
					print("Switching to Pure Exploit")
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, state):
		eps = self.getEpsilon()			
		FRAMES.next()

		if random.random() < eps and not TESTING:
			action = self.env.get_random_action()
			return action, True
		else:
			topo = np.array([state["topology"]])
			routes = np.array([state["routes"]])
			reach = np.array([state["reachability"]])
			prob = brain.predict_prob([topo, routes, reach])[0]
			action = self.env.get_random_action(p=prob)
			return action, False
	
	def train(self, state, action, reward, state_):
		def get_sample(memory, n):
			state, action, _, _  = memory[0]
			_, _, _, state_ = memory[n-1]

			return state, action, self.R, state_

		flow_id = action[0]
		switch_id = action[1]
		action_one_hot_encoded = np.zeros((self.action_shape_height, self.action_shape_width))
		action_one_hot_encoded[flow_id - 1][switch_id - 1] = 1

		self.memory.append( (state, action_one_hot_encoded, reward, state_) )

		self.R = ( self.R + reward * GAMMA_N ) / GAMMA

		if state_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				state, action, reward, state_ = get_sample(self.memory, n)
				brain.train_push(state, action, reward, state_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			state, action, reward, state_ = get_sample(self.memory, N_STEP_RETURN)
			brain.train_push(state, action, reward, state_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
class Environment(threading.Thread):
	# stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)
		self.num_instances = 0
		self.instance_iter = 0
		self.deviation = 0.0
		self.reset()
	
	def reset(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		self.stop_signal = False
		self.render = render
		self.env = gym.make(ENV)
		# self.env.__init__(topo_size=4, num_flows=1, topo_style='fat_tree', deterministic=True)
		self.env.__init__(topo_size=4, num_flows=1, topo_style='fat_tree')
		self.agent = Agent(eps_start, eps_end, eps_steps, self.env)
		self.time_begin = time.time()
		self.unique_id = INSTANCE_NUM.next()
		self.num_instances = self.num_instances + 1
		self.instance_iter = 0
		if DEBUG:
			print("INSTANCE NUMBER: %d" % self.unique_id)

	def runEpisode(self, hard_reset = False):
		state = self.env.reset()
		if DEBUG:
			print("Flow Details: %s" % str(self.env.flow_details))

		while True:         
			time.sleep(THREAD_DELAY)

			if self.render: 
				self.env.render()
			
			action, is_rand = self.agent.act(state)
			state_, reward, done, info = self.env.step(action)
			if DEBUG:
				print("Routes: %s" % str(state["routes"]))
				print("Action: %s, Random: %s" % (str(action), str(is_rand)))
				print("Reward: %s" % str(reward))

			if done:
				if DEBUG:
					print("DONE")
				state_ = None
				if not self.env.is_game_over:
					time_now = time.time()
					self.stop_signal = True
					self.instance_iter = self.instance_iter + 1
					instance_deviation = self.env.get_path_length_quality()
					self.deviation = self.deviation + instance_deviation
					if VERBOSE:
						print("TIME: %d, DEVIATION: %f" % ((time_now - self.time_begin), instance_deviation))
			
			if not TESTING:
				self.agent.train(state, action, reward, state_)

			state = state_

			if done or self.stop_signal:
				break

	def run(self):
		while True:
			if not TESTING:
				if self.unique_id > TRAINING_INSTANCE_LIMIT:
					self.stop_signal = True
					break
			else:
				if self.unique_id > TESTING_INSTANCE_LIMIT:
					self.stop_signal = True
					break
			if not self.stop_signal:
				self.runEpisode()
			else:
				if not TESTING:
					if self.instance_iter >= PER_INSTANCE_LIMIT:
						self.reset()
						self.runEpisode()
					else:
						self.stop_signal = False
						self.runEpisode()
				else:
					self.reset()
					self.runEpisode()

	def stop(self):
		self.stop_signal = True

#---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain.optimize()

	def stop(self):
		self.stop_signal = True

#-- main
FRAMES = itertools.count()
INSTANCE_NUM = itertools.count()
TRAINING_INSTANCE_LIMIT = 1000
PER_INSTANCE_LIMIT = 100
TESTING_INSTANCE_LIMIT = 1000
TESTING = False

_env = Environment(render=False, eps_start=0., eps_end=0.)
OBSERVATION_SPACE = _env.env.observation_space
ACTION_SPACE = _env.env.action_space
NULL_STATE = _env.env.get_null_state()
# STATE = env_test.env.observation_space.shape # 2D array shape with 0 or 1
# # print "State: %d, %d" % (STATE[0], STATE[1])
# ACTION = env_test.env.action_space # a tuple with non-zero inputs
# # print "Action: %d, %d" % (ACTION.high[0], ACTION.high[1])
# NO_STATE = np.zeros(STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
	o.start()

for e in envs:
	e.start()

for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()

training_instances = 0
training_deviation = 0.0
for e in envs:
	training_instances = training_instances + e.num_instances
	training_deviation = training_deviation + e.deviation
avg_training_deviation = training_deviation/(1.0*training_instances*PER_INSTANCE_LIMIT)

if VERBOSE:
	print("TRAINING PHASE ENDED.")

TESTING = True
INSTANCE_NUM = itertools.count()
envs = [Environment() for i in range(THREADS)]
test_instances = 0
test_deviation = 0.0
for e in envs:
	e.start()
for e in envs:
	e.join()
for e in envs:
	test_instances = test_instances + e.num_instances
	test_deviation = test_deviation + e.deviation

avg_test_deviation = test_deviation/(test_instances*1.0)
if VERBOSE:
	print("AVG TRAIN DEVIATION: %f" % avg_training_deviation)
	print("AVG TEST DEVIATION: %f"  % avg_test_deviation)
