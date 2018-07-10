#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import math
import random

# 3rd party modules
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SkynetEnv(gym.Env):
    metadata = {}

    def __init__(self):
    	self.num_destinations = 5
    	self.queue_length = 10
    	self.max_steps = 100
        self.observation_space = spaces.Box(0, self.num_destinations - 1, shape=(self.queue_length,))
        self.action_space = spaces.Discrete(self.num_destinations)
        self.state = None
        self.viewer = None

  	def step(self, action):
  		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
  		pkt_dst = self.state[0]
  		new_pkt = random.randint(0, self.num_destinations - 1)
  		self.state[0] = new_pkt
  		np.roll(self.state, -1)
  		reward = -1.0
  		if pkt_dst == action:
  			reward = 1.0
  		done = False
  		self.max_steps = self.max_steps - 1
  		if self.max_steps == 0:
  			done = True
  		return self.state, reward, done, {}

  	def render(self, mode='human'):
  		pass

  	def close(self):
  		pass

  	def reset(self):
  		self.state = np.random.randint(self.num_destinations, size=self.queue_length)
  		return self.state

  	def _reset(self):
  		self.state = np.random.randint(self.num_destinations, size=self.queue_length)
  		return self.state

  	def seed(self, seed=None):
  		pass