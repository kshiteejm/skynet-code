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

    def __init__(self, num_dests=5, q_len=10):
        self.__version__ = "0.1.0"
        self.num_destinations = num_dests
        self.queue_length = q_len
        self.max_steps = 100
        self.observation_space = spaces.Box(0, self.num_destinations - 1, shape=(self.queue_length,), dtype=np.int64)
        self.action_space = spaces.Discrete(self.num_destinations)
        self.state = None
        self.viewer = None

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        pkt_dst = state[0]
        new_pkt = random.randint(0, self.num_destinations - 1)
        state[0] = new_pkt
        self.state = np.roll(state, -1).tolist()
        reward = -1.0
        if pkt_dst == action:
            reward = 1.0
        done = False
        # self.max_steps = self.max_steps - 1
        if self.max_steps == 0:
            done = True
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        return

    def reset(self):
        self.state = np.random.randint(self.num_destinations, size=self.queue_length).tolist()
        return np.array(self.state)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed
