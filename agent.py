from __future__ import print_function

import logging
import itertools
import random

import numpy as np

from constants import EPS_START, EPS_END, EPS_STEPS, TESTING

class Agent:
    FRAMES = itertools.count()
    EXPLOIT = itertools.count()

    def __init__(self, env, brain, eps_start=EPS_START, eps_end=EPS_END, 
                eps_steps=EPS_STEPS, test=TESTING):
        self.env = env
        self.brain = brain
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps

        self.test = test

        self.gamma = brain.gamma
        self.gamma_n = brain.gamma_n
        self.n_step_return = brain.n_step_return

        self.memory = []    # used for n_step return
        self.R = 0.

        self.action_shape_height = int(self.env.action_space.high[0])
        self.action_shape_width = int(self.env.action_space.high[1])

    def getEpsilon(self):
        eps_ret = 0.0
        frames = next(Agent.FRAMES)
        if frames >= self.eps_steps:
            eps_ret = self.eps_end
        else:
            eps_ret = self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps    # linearly interpolate
        
        if not self.test and eps_ret == 0.0 and next(Agent.EXPLOIT) == 0:
            next(Agent.EXPLOIT)
            logging.info("Switching to Pure Exploit")

        return eps_ret

    def act(self, state):
        eps = self.getEpsilon()
        next(Agent.FRAMES)

        if random.random() < eps and not self.test:
            action = self.env.get_random_next_hop()
            return action, True
        else:
            logging.debug("AGENT: Next Hop Indices: %s", state["next_hop_indices"])
            # state = np.array([state])
            probabilities = self.brain.predict_prob(state)[0]
            action = self.env.get_random_next_hop(p=probabilities)
            return action, False
    
    def train(self, state, action, reward, state_):
        def get_sample(memory, n):
            state, action, _, _  = memory[0]
            _, _, _, state_ = memory[n-1]

            return state, action, self.R, state_

        next_hop, next_hop_len, next_hop_index = action
        action_one_hot_encoded = np.zeros(next_hop_len)
        action_one_hot_encoded[next_hop_index] = 1

        self.memory.append( (state, action_one_hot_encoded, reward, state_) )

        self.R = ( self.R + reward * self.gamma_n ) / self.gamma

        if state_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                state, action, reward, state_ = get_sample(self.memory, n)
                self.brain.train_push(state, action, reward, state_)

                self.R = ( self.R - self.memory[0][2] ) / self.gamma
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= self.n_step_return:
            state, action, reward, state_ = get_sample(self.memory, self.n_step_return)
            self.brain.train_push(state, action, reward, state_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)  
