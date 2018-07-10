#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import math
import random

# 3rd party modules
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SkynetEnv(gym.Env):

    def __init__(self):
    	self.num_destinations = 5
    	self.queue_length = 10
        self.observation_space = spaces.Box(1, self.num_destinations, shape=(self.queue_length,))
        self.action_space = spaces.Discrete(self.num_destinations)

  	def step(self, action):
  		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
  		state = self.state

