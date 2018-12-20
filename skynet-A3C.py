# borrowed from Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading
import gym_skynet

from keras.models import *
from keras.layers import *
from keras import backend as K

#-- constants
ENV = 'Skynet-v0'

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99
# GAMMA = 0.1
# GAMMA = 0.7

N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = 0.0
# EPS_START = 0.
# EPS_STOP  = 0.
EPS_STEPS = 12000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

	def _build_model(self):

		l_input = Input(batch_shape=(None, STATE[0]*STATE[1]))
		l_dense = Dense(16, activation='relu')(l_input)

		out_actions = Dense(ACTION.high[0]*ACTION.high[1], activation='softmax')(l_dense)
		out_value   = Dense(1, activation='linear')(l_dense)

		model = Model(inputs=[l_input], outputs=[out_actions, out_value])
		model._make_predict_function()	# have to initialize before threading

		return model

	def _build_graph(self, model):
		s_t = tf.placeholder(tf.float32, shape=(None, STATE[0]*STATE[1]))
		a_t = tf.placeholder(tf.float32, shape=(None, ACTION.high[0]*ACTION.high[1]))
		r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
		p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

		s = np.vstack(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.vstack(s_)
		s_mask = np.vstack(s_mask)
		# print s.shape
		# print a.shape
		# print a

		if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		# print r
		
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s.reshape(-1))
			self.train_queue[1].append(a.reshape(-1))
			# self.train_queue[1].append(ACTION.high[1]*(a[0]-1) + (a[1] - 1))
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(NO_STATE.reshape(-1))
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_.reshape(-1))
				self.train_queue[4].append(1.)

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return v

#---------
frames = 0
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

		self.memory = []	# used for n_step return
		self.R = 0.
		self.R_rand = 0.
		self.R_ = 0.
		self.num_steps = 0

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		global frames; frames = frames + 1

		if random.random() < eps:
			self.R_rand = self.R_rand + 1
			action = (random.randint(1, ACTION.high[0]), random.randint(1, ACTION.high[1]))
			while (s[action[0]-1][action[1]-1] != 0):
				action = (random.randint(1, ACTION.high[0]), random.randint(1, ACTION.high[1]))
			return action, True
		else:
			s_ = np.array([s.reshape(-1).tolist()])
			p = brain.predict_p(s_)[0]
			# print p

			# a = np.argmax(p)
			a = np.random.choice(ACTION.high[0]*ACTION.high[1], p=p)
			action = (a/ACTION.high[1] + 1, a%ACTION.high[1]+1)
			while (s[action[0]-1][action[1]-1] != 0):
				a = np.random.choice(ACTION.high[0]*ACTION.high[1], p=p)
				action = (a/ACTION.high[1] + 1, a%ACTION.high[1]+1)
			return action, False
	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros((ACTION.high[0], ACTION.high[1]))	# turn action into one-hot representation
		a_cats[a[0]-1][a[1]-1] = 1
		# a_cats = a_cats.reshape(-1)
		# print a_cats.shape

		self.memory.append( (s, a_cats, r, s_) )
		# self.memory.append( (s, a, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA
		self.R_ = self.R_ + r
		self.num_steps = self.num_steps + 1
		if self.num_steps >= 300:
			# print self.R_
			self.R_ = 0.
			self.num_steps = 0

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				brain.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)		

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			# print r
			brain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)

		self.render = render
		self.env = gym.make(ENV)
		self.env.__init__(topo_size=4, num_flows=10)
		self.agent = Agent(eps_start, eps_end, eps_steps)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		R_neg = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a, is_rand = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				# print s_
				s_ = None
				if r > 0:
					self.stop_signal = True
					global time_begin
					time_now = time.time()
					print "EXECUTION TIME: %d" % (time_now - time_begin)
					break

			self.agent.train(s, a, r, s_)

			s = s_
			if not is_rand:
				if r < -0.5:
					R_neg = R_neg + 1
					R += -1.0
				else:
					R += 1.0

			if done or self.stop_signal:
				break

		# print("Total R:", R)
		# print("Total R_neg:", R_neg)

	def run(self):
		while not self.stop_signal:
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
env_test = Environment(render=False, eps_start=0., eps_end=0.)
STATE = env_test.env.observation_space.shape # 2D array shape with 0 or 1
# print "State: %d, %d" % (STATE[0], STATE[1])
ACTION = env_test.env.action_space # a tuple with non-zero inputs
# print "Action: %d, %d" % (ACTION.high[0], ACTION.high[1])
NO_STATE = np.zeros(STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
	o.start()

time_begin = time.time()

for e in envs:
	e.start()

time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

print "SECOND INSTANCE"

envs = [Environment() for i in range(THREADS)]

time_begin = time.time()

for e in envs:
	e.start()

time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()

print("Training finished")

# env_test.start()
# time.sleep(30)
# env_test.stop()
# env_test.join()
