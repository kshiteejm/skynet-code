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

    def __init__(self, topo_size=2, num_flows=1):
        self.__version__ = "0.1.0"
       
        # static state
        # topology state
        self.topo_scale_factor = topo_size
        self.num_edge_switches = topo_size*topo_size
        self.num_agg_switches = topo_size
        self.num_core_switches = topo_size/2
        self.num_switches = self.num_edge_switches + self.num_agg_switches + self.num_core_switches
        self.num_links = self.num_edge_switches + self.num_agg_switches*self.num_core_switches
        self.completed_flows = []
        self.switch_switch_map = {}
        self.link_switch_map = {}
        self.switch_link_map = {}
        self._init_switch_switch_map()
        self._init_link_switch_map()
        self._init_switch_link_map()
        # flow traffic state
        self.num_flows = num_flows
        self.flow_details = {}
        self._init_flow_details()
        
        # observation space: #flows * #network links
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_links), dtype=np.uint8) 
        # action space: flow-id, network-link-id
        self.action_space = spaces.Box(low=np.array([1, 1]), high=np.array([self.num_flows, self.num_links]), dtype=np.uint8)

        # internal state for registering the network links that flows traverse so far
        self.state = np.zeros((self.num_flows, self.num_links), dtype=np.uint8)
        self.viewer = None

    # random endpoints for flows in the network
    def _init_flow_details(self):
        for flow_id in range(1, self.num_flows + 1):
            src_switch_id = random.randint(1, self.num_edge_switches)
            dst_switch_id = random.randint(1, self.num_edge_switches)
            while src_switch_id == dst_switch_id:
                dst_switch_id = random.randint(1, self.num_edge_switches)
            self.flow_details[flow_id] = [src_switch_id, dst_switch_id]

    # initialization of map from switch to switch neighbors
    def _init_switch_switch_map(self):
        for switch_id in range(1, self.num_switches + 1):
            self.switch_switch_map[switch_id] = []
        switch_id = 1
        # for edge - agg switches
        while switch_id <= self.num_edge_switches:
            # agg switch neighbor
            neighbor_id = (switch_id - 1)/self.topo_scale_factor + 1 + self.num_edge_switches
            self.switch_switch_map[switch_id].append(neighbor_id)
            self.switch_switch_map[neighbor_id].append(switch_id)
            switch_id = switch_id + 1
        # for agg - core switches
        while switch_id <= self.num_edge_switches+self.num_agg_switches:
            # core switch neighbors
            for neighbor_id in range(self.num_edge_switches+self.num_agg_switches+1, self.num_switches+1):
                self.switch_switch_map[switch_id].append(neighbor_id)
                self.switch_switch_map[neighbor_id].append(switch_id)
            switch_id = switch_id + 1

    # initialization of map from link to switches attached to that link
    def _init_link_switch_map(self):
        link_id = 1
        # for edge - agg links
        while link_id <= self.num_edge_switches:
            self.link_switch_map[link_id] = [link_id, self.num_edge_switches + (link_id - 1)/self.topo_scale_factor + 1]
            link_id = link_id + 1
        # for agg - core links
        base_link_id = link_id
        while link_id <= self.num_links:
            self.link_switch_map[link_id] = [base_link_id + (link_id-base_link_id)/self.num_core_switches, 
                    base_link_id + self.num_agg_switches + (link_id-base_link_id)%self.num_core_switches]
            link_id = link_id + 1

    # initialization of map from switch to links attached to that switch
    def _init_switch_link_map(self):
        for switch_id in range(1, self.num_switches + 1):
            self.switch_link_map[switch_id] = []
        for link_id in range(1, self.num_links + 1):
            for switch_id in self.link_switch_map[link_id]:
                self.switch_link_map[switch_id].append(link_id)

    def _get_neighbors(self, flow_id, switch_id):
        switch_links = self.switch_link_map[switch_id]
        active_links = [link_id for link_id in switch_links if self.state[flow_id-1][link_id-1] == 1]
        neighbors = set()
        for link_id in active_links:
            link_switches = self.link_switch_map[link_id]
            neighbors.add(link_switches[0])
            neighbors.add(link_switches[1])
        if switch_id in neighbors:
            neighbors.remove(switch_id)
        return neighbors
            
    def _connected_components(self, flow_id):
        connected_components = []
        is_cyclic = False
        switches = set(range(1, self.num_switches + 1))
        while switches:
            switch_id = switches.pop()
            group = {switch_id}
            queue = [(switch_id, -1)]
            while queue:
                switch_id, parent_id = queue.pop()
                neighbors = self._get_neighbors(flow_id, switch_id)
                for neighbor_id in neighbors:
                    if neighbor_id in group:
                        if neighbor_id != parent_id:
                            is_cyclic = True
                    else:
                        switches.remove(neighbor_id)
                        group.add(neighbor_id)
                        queue.append((neighbor_id, switch_id))
            connected_components.append(group)
        return connected_components, is_cyclic

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        flow_id, link_id = action
        done = False
        reward = -0.01
        flow_switches = self.flow_details[flow_id]
        # if flow_id in self.completed_flows or state[flow_id-1][link_id-1] == 1: # this should not be happening
        self.state[flow_id-1][link_id-1] = 1
        connected_components, is_cyclic = self._connected_components(flow_id)
        if is_cyclic:
            reward = -1.0
            done = True
        else:
            for group in connected_components:
                if flow_switches[0] in group and flow_switches[1] in group:
                    self.completed_flows.append(flow_id)
                    reward = 1.0
                    if len(self.completed_flows) == self.num_flows:
                        done = True
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        return

    def reset(self):
        self.state = np.zeros((self.num_flows, self.num_links), dtype=np.uint8)
        return np.array(self.state)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed
