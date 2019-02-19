from __future__ import print_function

import logging
import time
import itertools
import threading

import gym

from constants import EPS_START, EPS_END, EPS_STEPS, PER_INSTANCE_LIMIT, \
    THREAD_DELAY, ENV, TRAINING_INSTANCE_LIMIT, TESTING_INSTANCE_LIMIT, \
    TESTING

from agent import Agent

class Environment(threading.Thread):
    INSTANCE_NUM = itertools.count()

    def __init__(self, brain, num_flows, num_switches, topo,
                node_features=None, render=False, eps_start=EPS_START,
                eps_end=EPS_END, eps_steps=EPS_STEPS, thread_delay=THREAD_DELAY,
                per_instance_limit=PER_INSTANCE_LIMIT, 
                training_instance_limit=TRAINING_INSTANCE_LIMIT, 
                testing_instance_limit=TESTING_INSTANCE_LIMIT,
                test=TESTING):

        threading.Thread.__init__(self)
        self.instance_iter = 0
        self.deviation = 0.0
        self.num_instances = 0

        self.node_features = node_features
        self.brain = brain

        self.num_flows = num_flows 
        self.num_switches = num_switches 
        self.topo = topo
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.thread_delay = thread_delay

        self.per_instance_limit = per_instance_limit
        self.training_instance_limit = training_instance_limit
        self.testing_instance_limit = testing_instance_limit

        self.test = test

        self.stop_signal = False

        self.reset()
    
    def reset(self, render=False, eps_start=None, eps_end=None, eps_steps=None):
        if eps_start is None: 
            eps_start = self.eps_start
        if eps_end is None: 
            eps_end = self.eps_end
        if eps_steps is None: 
            eps_steps = self.eps_steps

        self.stop_signal = False
        self.render = render
        self.env = gym.make(ENV)
        self.env.__init__(topo_size=self.num_switches, num_flows=self.num_flows, topo_style=self.topo)
        self.agent = Agent(self.env, self.brain, eps_start=eps_start, eps_end=eps_end, 
                            eps_steps=eps_steps, test=self.test)
        
        self.topology = self.env.state["topology"]

        self.time_begin = time.time()
        self.unique_id = next(Environment.INSTANCE_NUM)
        self.num_instances += 1
        self.instance_iter = 0
        logging.debug("Instance Number: %d", self.unique_id)

    def runEpisode(self):
        state = self.env.reset()
        logging.debug("Flow Details: %s", str(self.env.flow_details))

        while True:         
            time.sleep(self.thread_delay)

            if self.render: 
                self.env.render()
            
            action, is_rand = self.agent.act(state)
            state_, reward, done, info = self.env.step(action)
            
            logging.debug("Raw Node Features List Shape:  %s", str(state["raw_node_feature_list"].shape))
            logging.debug("Action: %s, Random: %s", str(action), str(is_rand))
            logging.debug("Reward: %s", str(reward))
            
            if done:
                logging.debug("DONE")
                # state_ = None
                if not self.env.is_game_over:
                    time_now = time.time()
                    self.stop_signal = True
                    self.instance_iter = self.instance_iter + 1
                    instance_deviation = self.env.get_path_length_quality()
                    self.deviation = self.deviation + instance_deviation
                    logging.debug("TIME: %d, DEVIATION: %f", (time_now - self.time_begin), instance_deviation)
            
            if not self.test:
                self.agent.train(state, action, reward, state_)

            state = state_

            if done or self.stop_signal:
                break

    def run(self):
        while True:
            if not self.test:
                if self.unique_id > self.training_instance_limit:
                    self.stop_signal = True
                    break
            else:
                if self.unique_id > self.testing_instance_limit:
                    self.stop_signal = True
                    break
            if not self.stop_signal:
                self.runEpisode()
            else:
                if not self.test:
                    if self.instance_iter >= self.per_instance_limit:
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

    def getState(self):
        return self.env.state

    def getNodeFeatures(self):
        return self.env.state["raw_node_feature_list"]        