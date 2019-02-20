#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import sys
sys.path.append("../../")

import logging

import math
import random

# 3rd party modules
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from constants import MIN_GRAPH_SIZE, MAX_GRAPH_SIZE, MIN_FLOWS, MAX_FLOWS

NEG_INF = -100.0
POS_INF = 100.0
# MIN_GRAPH_SIZE = 5
# MAX_GRAPH_SIZE = 60
MAX_NUM_SWITCHES = MAX_GRAPH_SIZE*MAX_GRAPH_SIZE*5//4
# MIN_FLOWS = 1
# MAX_FLOWS = 100

class SkynetEnv(gym.Env):

    def __init__(self, topo_size=2, num_flows=1, topo_style='fat_tree', deterministic=False, raw_node_feature_list=None):
        self.__version__ = "0.1.0"
       
        self.is_game_over = False
        self.topo_scale_factor = topo_size

        # topology state
        if topo_style == 'std_dcn':
            self.num_edge_switches = self.topo_scale_factor*self.topo_scale_factor
            self.num_agg_switches = self.topo_scale_factor
            self.num_core_switches = self.topo_scale_factor // 2
            self.num_switches = self.num_edge_switches + self.num_agg_switches + self.num_core_switches
            self.num_links = self.num_edge_switches + self.num_agg_switches*self.num_core_switches

        if topo_style == 'fat_tree':
            self.topo_scale_factor = topo_size
            self.num_edge_switches = (self.topo_scale_factor * self.topo_scale_factor) // 2
            self.num_agg_switches = (self.topo_scale_factor * self.topo_scale_factor) // 2
            self.num_core_switches = (self.topo_scale_factor // 2) * (self.topo_scale_factor // 2)
            self.num_switches = self.num_edge_switches + self.num_agg_switches + self.num_core_switches
            self.num_links = (2 * self.num_agg_switches * (self.topo_scale_factor // 2)) + (self.num_edge_switches * self.topo_scale_factor) // 2

        self.switch_switch_map = {}
        # self.link_switch_map = {}
        # self.switch_link_map = {}
        self._init_switch_switch_map(topo_style)
        # self._init_link_switch_map(topo_style)
        # self._init_switch_link_map(topo_style)

        # flow traffic state
        self.num_flows = int(num_flows)
        self.flow_details = {}
        self.flow_switch_map = {}
        self.completed_flows = []
        self.incomplete_flows = []
        self._init_flow_details(deterministic=deterministic)

        # isolation (vertex isolation for now) policies
        self.num_isolation_policies = int(self.num_flows / 4)
        self.isolation = np.zeros((MAX_FLOWS, MAX_FLOWS))
        self._init_isolation_policies(deterministic=deterministic)

        # visited, can_visit
        self.raw_node_feature_size = 2 

        # # node (per-switch) features
        # self.raw_node_feature_list = raw_node_feature_list
        # self.next_hop_features = np.array([])
        # self.next_hop_details = []

        # node (per-switch, per-node) features
        self.raw_node_feature_list = np.zeros((self.num_flows, self.num_switches, self.raw_node_feature_size))
        self.next_hop_indices = np.array([])
        
        # observation space:
        self.observation_space = spaces.Dict(dict(
            topology=spaces.Box(low=0, high=1, shape=(self.num_switches, self.num_switches)),
            routes=spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_switches)),
            reachability=spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_switches)),
            isolation=spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_flows)),
            raw_node_feature_list=spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_switches, self.raw_node_feature_size))
        ))

        # action space:
        self.action_space = spaces.Box(low=np.array([1, 1]), high=np.array([self.num_flows, self.num_switches]))

        # state:
        self.state = dict(
            topology=np.zeros((self.num_switches, self.num_switches)),
            routes=np.zeros((self.num_flows, self.num_switches)),
            reachability=np.zeros((MAX_FLOWS, MAX_NUM_SWITCHES)),
            isolation=np.zeros((MAX_FLOWS, MAX_FLOWS)),
            raw_node_feature_list=np.zeros((self.num_flows, self.num_switches, self.raw_node_feature_size)),
            # next_hop_features=np.array([]),
            next_hop_indices=np.array([])
        )

        self._init_state()

        # # observation space: #flows * #network links
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_flows, self.num_links), dtype=np.uint8) 

        # # action space: flow-id, network-link-id
        # self.action_space = spaces.Box(low=np.array([1, 1]), high=np.array([self.num_flows, self.num_links]), dtype=np.uint8)

        # # internal state for registering the network links that flows traverse so far
        # self.state = np.zeros((self.num_flows, self.num_links), dtype=np.uint8)

        self.viewer = None

    def _init_state(self):
        self.state = dict(
            topology=np.zeros((self.num_switches, self.num_switches)),
            routes=np.zeros((self.num_flows, self.num_switches)),
            reachability=np.zeros((MAX_FLOWS, MAX_NUM_SWITCHES)),
            isolation=np.zeros((MAX_FLOWS, MAX_FLOWS)),
            raw_node_feature_list=np.zeros((self.num_flows, self.num_switches, self.raw_node_feature_size)),
            # next_hop_features=np.array([]),
            next_hop_indices=np.array([])
        )
        
        topology = self.state["topology"]
        routes = self.state["routes"]
        reachability = self.state["reachability"]
        raw_node_feature_list = self.state["raw_node_feature_list"]
        next_hop_indices = self.state["next_hop_indices"]
        isolation = self.state["isolation"]

        # initialize topology
        for src_switch_id in self.switch_switch_map:
            for dst_switch_id in self.switch_switch_map[src_switch_id]:
                topology[src_switch_id-1][dst_switch_id-1] = 1
        
        # initialize routes, reachability
        # each flow route contains source switch to begin with
        # print(self.flow_details)
        for flow_id in self.flow_details:
            src_switch_id, dst_switch_id = self.flow_details[flow_id]
            routes[flow_id-1][src_switch_id-1] = 1
            reachability[flow_id-1][src_switch_id-1] = 1
            reachability[flow_id-1][dst_switch_id-1] = 1
        
        # propagate isolation
        isolation = np.array(self.isolation)
        
        # update per-flow raw node features
        self.raw_node_feature_list = self._generate_raw_node_feature_list()
        raw_node_feature_list = np.array(self.raw_node_feature_list)

        # update next hop indices
        self.next_hop_indices = self.get_next_hop_indices()
        next_hop_indices = np.array(self.next_hop_indices)

        # # update next_hop_features, next_hop_details
        # if self.raw_node_feature_list is not None:
        #     next_hop_features, next_hop_details = self.get_next_hop_features()
        #     self.next_hop_features = next_hop_features
        #     self.state["next_hop_features"] = next_hop_features
        #     self.next_hop_details = next_hop_details
    
    def get_null_state(self):
        topology = self.state["topology"]
        routes = self.state["routes"]
        reachability = self.state["reachability"]
        isolation = self.state["isolation"]
        raw_node_feature_list = self.state["raw_node_feature_list"]
        # next_hop_indices = self.state["next_hop_indices"]
        return dict(topology=np.array(topology), routes=np.zeros(routes.shape), reachability=np.zeros(reachability.shape), isolation=np.zeros(isolation.shape), raw_node_feature_list=np.zeros(raw_node_feature_list.shape), next_hop_indices=np.array([]))
    
    # random endpoints for flows in the network
    def _init_flow_details(self, initialize=True, deterministic=False):
        self.completed_flows = []
        self.incomplete_flows = list(range(1, self.num_flows + 1))
        self.is_game_over = False
        for flow_id in range(1, self.num_flows + 1):
            if deterministic:
                src_switch_id = (flow_id%self.num_edge_switches) + 1
                dst_switch_id = ((src_switch_id + self.num_edge_switches//2)%self.num_edge_switches) + 1
                self.flow_details[flow_id] = [src_switch_id, dst_switch_id]
                self.flow_switch_map[flow_id] = [src_switch_id]
            else:
                if initialize:
                    src_switch_id = random.randint(1, self.num_edge_switches)
                    dst_switch_id = random.randint(1, self.num_edge_switches)
                    while src_switch_id == dst_switch_id:
                        dst_switch_id = random.randint(1, self.num_edge_switches)
                    self.flow_details[flow_id] = [src_switch_id, dst_switch_id]
                    self.flow_switch_map[flow_id] = [src_switch_id]
                else:
                    src_switch_id, dst_switch_id = self.flow_details[flow_id]
                    self.flow_switch_map[flow_id] = [src_switch_id]
        # print "flow details: %s" % str(self.flow_details)
    
    def _init_isolation_policies(self, initialize=True, deterministic=False):
        for i in range(self.num_isolation_policies):
            if deterministic:
                first_flow_id = (i%self.num_flows) + 1
                second_flow_id = ((i + self.num_flows//2)%self.num_flows) + 1
                self.isolation[first_flow_id-1][second_flow_id-1] = 1
                self.isolation[second_flow_id-1][first_flow_id-1] = 1
            else:
                if initialize:
                    first_flow_id = random.randint(1, self.num_flows)
                    second_flow_id = random.randint(1, self.num_flows)
                    while second_flow_id == first_flow_id:
                        second_flow_id = random.randint(1, self.num_flows)
                    self.isolation[first_flow_id-1][second_flow_id-1] = 1
                    self.isolation[second_flow_id-1][first_flow_id-1] = 1

    # initialization of map from switch to switch neighbors
    def _init_switch_switch_map(self, topo_style='fat_tree'):
        for switch_id in range(1, self.num_switches + 1):
            self.switch_switch_map[switch_id] = []

        if topo_style == 'std_dcn':
            switch_id = 1
            # for edge - agg switches
            while switch_id <= self.num_edge_switches:
                # agg switch neighbor
                neighbor_id = (switch_id - 1)//self.topo_scale_factor + 1 + self.num_edge_switches
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
        
        if topo_style == 'fat_tree':
            # print "total edge switches: %d" % self.num_edge_switches
            # print "scale factor: %d" % self.topo_scale_factor
            switch_id = 1
            # for edge - agg switches
            while switch_id <= self.num_edge_switches:
                edge_switches = list(range(switch_id, switch_id+(self.topo_scale_factor // 2)))
                agg_switches = list(range(switch_id+self.num_edge_switches, switch_id+self.num_edge_switches+self.topo_scale_factor//2))
                for edge_switch_id in edge_switches:
                    for agg_switch_id in agg_switches:
                        # print "edge_switch_id: %d, agg_switch_id: %d" % (edge_switch_id, agg_switch_id)
                        self.switch_switch_map[edge_switch_id].append(agg_switch_id)
                        self.switch_switch_map[agg_switch_id].append(edge_switch_id)
                switch_id = switch_id + self.topo_scale_factor//2
            # for agg - core switches
            core_switches = list(range(self.num_edge_switches+self.num_agg_switches+1, self.num_switches+1))
            while switch_id <= self.num_edge_switches+self.num_agg_switches:
                for core_switch_id in core_switches:
                    agg_switch_id = switch_id + (core_switch_id - core_switches[0])//(self.topo_scale_factor//2)
                    # print "agg_switch_id: %d, core_switch_id: %d" % (agg_switch_id, core_switch_id)
                    self.switch_switch_map[agg_switch_id].append(core_switch_id)
                    self.switch_switch_map[core_switch_id].append(agg_switch_id)
                switch_id = switch_id + self.topo_scale_factor//2

        # print "switch switch map: %s" % str(self.switch_switch_map)

    # # initialization of map from link to switches attached to that link
    # def _init_link_switch_map(self, topo_style='std_dcn'):
    #     link_id = 1
    #     # for edge - agg links
    #     while link_id <= self.num_edge_switches:
    #         self.link_switch_map[link_id] = [link_id, self.num_edge_switches + (link_id - 1)/self.topo_scale_factor + 1]
    #         link_id = link_id + 1
    #     # for agg - core links
    #     base_link_id = link_id
    #     while link_id <= self.num_links:
    #         self.link_switch_map[link_id] = [base_link_id + (link_id-base_link_id)/self.num_core_switches, 
    #                 base_link_id + self.num_agg_switches + (link_id-base_link_id)%self.num_core_switches]
    #         link_id = link_id + 1
    #     # print "link switch map: %s" % str(self.link_switch_map)

    # # initialization of map from switch to links attached to that switch
    # def _init_switch_link_map(self, topo_style='std_dcn'):
    #     for switch_id in range(1, self.num_switches + 1):
    #         self.switch_link_map[switch_id] = []
    #     for link_id in range(1, self.num_links + 1):
    #         for switch_id in self.link_switch_map[link_id]:
    #             self.switch_link_map[switch_id].append(link_id)

    def _generate_raw_node_feature_list(self):
        routes = self.state["routes"]
        raw_node_feature_list = np.zeros((self.num_flows, self.num_switches, self.raw_node_feature_size))
        # all traversed switches for each flow are marked as visited and cannot visit
        for flow_id in range(1, self.num_flows+1):
            flow_route = routes[flow_id-1]
            for switch_id in range(1, len(flow_route)+1):
                raw_node_feature_list[flow_id-1][switch_id-1][0] = flow_route[switch_id-1]
                raw_node_feature_list[flow_id-1][switch_id-1][1] = flow_route[switch_id-1]
        # isolation related node featurization
        for flow_id_1 in range(1, self.num_flows+1):
            for flow_id_2 in range(1, self.num_flows+1):
                if self.isolation[flow_id_1-1][flow_id_2-1] == 1:
                    routes_1 = routes[flow_id_1-1]
                    routes_2 = routes[flow_id_2-1]
                    src_switch_id_1, dst_switch_id_1 = self.flow_details[flow_id]
                    src_switch_id_2, dst_switch_id_2 = self.flow_details[flow_id]
                    # flow 1 cannot visit flow 2 switches and vice versa except their src/dst
                    for switch_id in range(1, len(routes_2)+1):
                        if switch_id == src_switch_id_2 or switch_id == dst_switch_id_2:
                            continue
                        raw_node_feature_list[flow_id_1-1][switch_id-1] = routes_2[switch_id-1]
                    for switch_id in range(1, len(routes_1)+1):
                        if switch_id == src_switch_id_1 or switch_id == dst_switch_id_1:
                            continue
                        raw_node_feature_list[flow_id_2-1][switch_id-1] = routes_1[switch_id-1]
        return raw_node_feature_list

    def _get_neighbors(self, flow_id, root_switch_id, visited=True):
        routes = self.state["routes"]
        next_switches = self.switch_switch_map[root_switch_id]
        neighbors = set()
        for switch_id in next_switches:
            if visited:
                if routes[flow_id-1][switch_id-1] == 1:
                    neighbors.add(switch_id)
            else:
                if routes[flow_id-1][switch_id-1] == 0:
                    neighbors.add(switch_id)
        if root_switch_id in neighbors:
            neighbors.remove(root_switch_id)
        return neighbors

    # def _get_neighbors(self, flow_id, root_switch_id, visited=True):
    #     switch_links = self.switch_link_map[root_switch_id]
    #     if visited:
    #         active_links = [link_id for link_id in switch_links if self.state[flow_id-1][link_id-1] == 1]
    #     else:
    #         active_links = [link_id for link_id in switch_links if self.state[flow_id-1][link_id-1] == 0]
    #     neighbors = set()
    #     for link_id in active_links:
    #         link_switches = self.link_switch_map[link_id]
    #         for switch_id in link_switches:
    #             if visited:
    #                 if switch_id in self.flow_switch_map[flow_id]:
    #                     neighbors.add(switch_id)
    #             else:
    #                 if switch_id not in self.flow_switch_map[flow_id]:
    #                     neighbors.add(switch_id)
    #     if root_switch_id in neighbors:
    #         neighbors.remove(root_switch_id)
    #     return neighbors
            
    def _connected_components(self, flow_id, visited=True):
        connected_components = []
        is_cyclic = False
        if visited:
            switches = set(self.flow_switch_map[flow_id])
        else:
            switches = set(range(1, self.num_switches + 1)).difference(self.flow_switch_map[flow_id])
        # print self.switch_switch_map
        # print "non-visited switches for flow %d: %s" % (flow_id, str(switches))
        while switches:
            switch_id = switches.pop()
            group = {switch_id}
            queue = [(switch_id, -1)]
            while queue:
                switch_id, parent_id = queue.pop()
                neighbors = self._get_neighbors(flow_id, switch_id, visited)
                # print "non-visited neighbors of switch %d: %s" % (switch_id, neighbors)
                for neighbor_id in neighbors:
                    if neighbor_id in group:
                        if neighbor_id != parent_id:
                            is_cyclic = True
                    else:
                        # if neighbor_id in switches:
                        switches.remove(neighbor_id)
                        group.add(neighbor_id)
                        queue.append((neighbor_id, switch_id))
            connected_components.append(group)
        return connected_components, is_cyclic

    # def get_random_action(self):
    #     random_flow_index = random.randint(1, len(self.incomplete_flows))
    #     random_flow_id = self.incomplete_flows[random_flow_index - 1]
    #     recent_flow_switch = self.flow_switch_map[random_flow_id][-1]
    #     all_next_link_ids = self.switch_link_map[recent_flow_switch]
    #     filtered_next_link_ids = [link_id for link_id in all_next_link_ids if self.state[random_flow_id - 1][link_id - 1] == 0]
    #     random_link_index = random.randint(1, len(filtered_next_link_ids))
    #     random_link_id = filtered_next_link_ids[random_link_index - 1]
    #     action = (random_flow_id, random_link_id)
    #     return action
    
    def _is_edge_switch(self, switch_id):
        if switch_id in range(1, self.num_edge_switches+1):
            return True
        else:
            return False
    
    def get_next_hop_indices(self):
        next_hop_indices = []
        for i in range(self.num_flows):
            next_hop_indices.append([])
        routes = self.state["routes"]
        for flow_id in self.incomplete_flows:
            recent_switch_id = self.flow_switch_map[flow_id][-1]
            next_switch_ids = self.switch_switch_map[recent_switch_id]
            filtered_next_switch_ids = [switch_id for switch_id in next_switch_ids if routes[flow_id - 1][switch_id - 1] == 0]
            _, dst_switch_id = self.flow_details[flow_id]
            connected_components, _ = self._connected_components(flow_id, visited=False)
            reachable_to_dst_next_switch_ids = []
            for group in connected_components:
                if dst_switch_id in group:
                    for switch_id in filtered_next_switch_ids:
                        if switch_id in group:
                            reachable_to_dst_next_switch_ids.append(switch_id)
            for switch_id in reachable_to_dst_next_switch_ids:
                next_hop_indices[flow_id-1].append(switch_id-1)
        return np.array(next_hop_indices)

    def get_next_hop_features(self):
        routes = self.state["routes"]
        raw_node_feature_list = self.raw_node_feature_list
        next_hop_features = []
        next_hop_details = []
        for flow_id in self.incomplete_flows:
            recent_switch_id = self.flow_switch_map[flow_id][-1]
            next_switch_ids = self.switch_switch_map[recent_switch_id]
            filtered_next_switch_ids = [switch_id for switch_id in next_switch_ids if routes[flow_id - 1][switch_id - 1] == 0]
            src_switch_id, dst_switch_id = self.flow_details[flow_id]
            connected_components, is_cyclic = self._connected_components(flow_id, visited=False)
            reachable_to_dst_next_switch_ids = []
            for group in connected_components:
                if dst_switch_id in group:
                    for switch_id in filtered_next_switch_ids:
                        if switch_id in group:
                            reachable_to_dst_next_switch_ids.append(switch_id)
            for switch_id in reachable_to_dst_next_switch_ids:
                next_hop_features.append(np.concatenate((raw_node_feature_list[switch_id-1], raw_node_feature_list[recent_switch_id-1], raw_node_feature_list[dst_switch_id-1], raw_node_feature_list[src_switch_id-1])))
                next_hop_details.append((flow_id, switch_id))
            logging.debug("flow-id: %s, src: %s, dst: %s", flow_id, src_switch_id, dst_switch_id)
            logging.debug("nxt-hop-details: %s", next_hop_details)
        return np.array(next_hop_features), next_hop_details

    # get a next hop link for an incomplete flow adhering to a particular probability distribution
    def get_random_action(self, p=None):
        routes = self.state["routes"]
        masked_p = np.zeros((self.num_flows, self.num_switches))
        for flow_id in self.incomplete_flows:
            recent_switch_id = self.flow_switch_map[flow_id][-1]
            next_switch_ids = self.switch_switch_map[recent_switch_id]
            filtered_next_switch_ids = [switch_id for switch_id in next_switch_ids if routes[flow_id - 1][switch_id - 1] == 0]
            src_switch_id, dst_switch_id = self.flow_details[flow_id]
            connected_components, is_cyclic = self._connected_components(flow_id, visited=False)
            reachable_to_dst_next_switch_ids = []
            for group in connected_components:
                if dst_switch_id in group:
                    for switch_id in filtered_next_switch_ids:
                        if switch_id in group:
                            reachable_to_dst_next_switch_ids.append(switch_id)
            # all_next_switch_ids = self.switch_switch_map[recent_flow_switch_id]
            # filtered_next_switch_ids = [switch_id for switch_id in all_next_switch_ids if switch_id not in self.flow_switch_map[flow_id]]
            for switch_id in reachable_to_dst_next_switch_ids:
                if p is None:
                    masked_p[flow_id - 1][switch_id - 1] = 1.0/float(len(reachable_to_dst_next_switch_ids))
                else:
                    masked_p[flow_id - 1][switch_id - 1] = p[flow_id - 1][switch_id - 1] + 1e-7
            if len(reachable_to_dst_next_switch_ids) == 0:
                logging.info("Flow %d from %d to %d has no next hop from %d", flow_id, src_switch_id, dst_switch_id, recent_switch_id)
        flow_p = np.sum(masked_p, axis=1)
        flow_p_sum = np.sum(flow_p)
        if flow_p_sum == 0:
            logging.info("No Viable Action Exception")
            return (-1 , -1)
        flow_p = [p/flow_p_sum for p in flow_p]
        random_flow_id = np.random.choice(list(range(1, self.num_flows + 1)), p=flow_p)
        switch_p = masked_p[random_flow_id - 1]
        switch_p_sum = np.sum(switch_p)
        if switch_p_sum == 0:
            logging.info("Flow Not Viable Exception")
            return (random_flow_id, -1)
        switch_p = [p/switch_p_sum for p in switch_p]
        random_switch_id = np.random.choice(list(range(1, self.num_switches + 1)), p=switch_p)
        return (random_flow_id, random_switch_id)

    def get_random_next_hop(self, p=None):
        next_hop_details = []
        for flow_id in range(1, len(self.next_hop_indices)+1):
            for switch_id in self.next_hop_indices[flow_id-1]:
                next_hop_details.append((flow_id, switch_id))
        random_next_hop_index = np.random.choice(list(range(0, len(next_hop_details))), p=p)
        return next_hop_details[random_next_hop_index], len(next_hop_details), random_next_hop_index

    # # get a next hop link for an incomplete flow adhering to a particular probability distribution
    # def get_random_action(self, p=None):
    #     masked_p = np.zeros((self.num_flows, self.num_links))
    #     for flow_id in self.incomplete_flows:
    #         recent_flow_switch_id = self.flow_switch_map[flow_id][-1]
    #         all_next_link_ids = self.switch_link_map[recent_flow_switch_id]
    #         filtered_next_link_ids = [link_id for link_id in all_next_link_ids if self.state[flow_id - 1][link_id - 1] == 0]
    #         src_switch_id, dst_switch_id = self.flow_details[flow_id]
    #         connected_components, is_cyclic = self._connected_components(flow_id, visited=False)
    #         reachable_to_dst_next_link_ids = []
    #         for group in connected_components:
    #             if dst_switch_id in group:
    #                 for link_id in filtered_next_link_ids:
    #                     link_src_switch_id, link_dst_switch_id = self.link_switch_map[link_id]
    #                     other_switch_id = link_src_switch_id if link_dst_switch_id == recent_flow_switch_id else link_dst_switch_id
    #                     if other_switch_id in group:
    #                         reachable_to_dst_next_link_ids.append(link_id)
    #         # all_next_switch_ids = self.switch_switch_map[recent_flow_switch_id]
    #         # filtered_next_switch_ids = [switch_id for switch_id in all_next_switch_ids if switch_id not in self.flow_switch_map[flow_id]]
    #         for link_id in reachable_to_dst_next_link_ids:
    #             if p is None:
    #                 masked_p[flow_id - 1][link_id - 1] = 1.0/float(len(reachable_to_dst_next_link_ids))
    #             else:
    #                 masked_p[flow_id - 1][link_id - 1] = p[flow_id - 1][link_id - 1] + 1e-7
    #         if len(reachable_to_dst_next_link_ids) == 0:
    #             print "Flow %d from %d to %d has no next hop from %d" % (flow_id, src_switch_id, dst_switch_id, recent_flow_switch_id)
    #     flow_p = np.sum(masked_p, axis=1)
    #     flow_p_sum = np.sum(flow_p)
    #     if flow_p_sum == 0:
    #         print "No Viable Action Exception"
    #         return (-1 , -1)
    #     flow_p = [p/flow_p_sum for p in flow_p]
    #     random_flow_id = np.random.choice(range(1, self.num_flows + 1), p=flow_p)
    #     link_p = masked_p[random_flow_id - 1]
    #     link_p_sum = np.sum(link_p)
    #     if link_p_sum == 0:
    #         print "Flow Not Viable Exception"
    #         return (random_flow_id, -1)
    #     link_p = [p/link_p_sum for p in link_p]
    #     random_link_id = np.random.choice(range(1, self.num_links + 1), p=link_p)
    #     return (random_flow_id, random_link_id)

    def get_avg_flow_path_length(self):
        sum_flow_path_len = 0.0
        for flow_id in self.flow_switch_map:
            sum_flow_path_len = sum_flow_path_len + len(self.flow_switch_map[flow_id])
        return (sum_flow_path_len)/(self.num_flows)

    def get_path_length_quality(self):
        quality = 0.0
        for flow_id in self.flow_details:
            src_switch_id, dst_switch_id = self.flow_details[flow_id]
            src_pod_id = (src_switch_id - 1)//(self.topo_scale_factor//2)
            dst_pod_id = (dst_switch_id - 1)//(self.topo_scale_factor//2)
            shortest_path_len = 2
            if src_pod_id == dst_pod_id:
                shortest_path_len = 3
            else:
                shortest_path_len = 5
            # print "flow-path: %s" % str(self.flow_switch_map[flow_id])
            flow_path_len = len(self.flow_switch_map[flow_id])
            quality = quality + flow_path_len - shortest_path_len
        return quality
    
    def is_isolation_violated(self, flow_id_1, nxt_switch_id):
        violated = False
        routes = self.state["routes"]
        for flow_id_2 in range(1, self.num_flows+1):
            if self.isolation[flow_id_1-1][flow_id_2-1] == 1:
                src_switch_id_2, dst_switch_id_2 = self.flow_details[flow_id_2]
                if nxt_switch_id == src_switch_id_2 or nxt_switch_id == dst_switch_id_2:
                    continue
                routes_2 = routes[flow_id_2-1]
                if routes_2[nxt_switch_id-1] == 1:
                    violated = True
                    break
        return violated

    def step(self, action):
        topology = self.state["topology"]
        routes = self.state["routes"]
        reachability = self.state["reachability"]
        isolation = self.state["isolation"]
        raw_node_feature_list = self.state["raw_node_feature_list"]
        next_hop_indices = self.state["next_hop_indices"]

        # get all necessary information before updating network state
        (flow_id, nxt_switch_id), next_hops_len, chosen_next_hop_index = action
        done = False
        reward = -1
        src_switch_id, dst_switch_id = self.flow_details[flow_id]
        connected_components, is_cyclic = self._connected_components(flow_id, visited=False)

        if self.is_isolation_violated(flow_id, nxt_switch_id) == True:
            reward = NEG_INF
            self.incomplete_flows.remove(flow_id)
            self.completed_flows.append(flow_id)
            self.is_game_over = True
            logging.info("GAME OVER")
        else:
            # get rewards and check game over conditions
            # check if the flow reaches desired destination
            if nxt_switch_id == dst_switch_id:
                # reward = POS_INF
                self.incomplete_flows.remove(flow_id)
                self.completed_flows.append(flow_id)
            else:
                # # check if the flow reaches wrong edge switch or there is a cycle
                # if self._is_edge_switch(nxt_switch_id) or nxt_switch_id in self.flow_switch_map[flow_id]:
                # check if next switch causes a cycle
                if nxt_switch_id in self.flow_switch_map[flow_id]:
                    reward = NEG_INF
                    self.incomplete_flows.remove(flow_id)
                    self.completed_flows.append(flow_id)
                    self.is_game_over = True
                    logging.info("GAME OVER")
                # check if the flow cannot reach the destination
                else:
                    for group in connected_components:
                        if dst_switch_id in group and nxt_switch_id not in group:
                            reward = NEG_INF
                            self.incomplete_flows.remove(flow_id)
                            self.completed_flows.append(flow_id)
                            self.is_game_over = True
                            logging.info("GAME OVER")
        
        # update all network state
        # update the switches visited by flow - valid if action is one of next hop link
        self.flow_switch_map[flow_id].append(nxt_switch_id)
        # if flow_id in self.completed_flows or state[flow_id-1][link_id-1] == 1: # this should not be happening
        routes[flow_id-1][nxt_switch_id-1] = 1
        self.raw_node_feature_list = self._generate_raw_node_feature_list()
        raw_node_feature_list = np.array(self.raw_node_feature_list)
        # next_hop_features, next_hop_details = self.get_next_hop_features()
        # self.next_hop_features = next_hop_features
        # self.state["next_hop_features"] = next_hop_features
        # self.next_hop_details = next_hop_details
        self.next_hop_indices = self.get_next_hop_indices()
        next_hop_indices = np.array(self.get_next_hop_indices())
        logging.debug("Next Hop Indices: %s", next_hop_indices)

        if len(self.completed_flows) == self.num_flows or self.is_game_over:
            done = True
        
        return dict(topology=np.array(topology), routes=np.array(routes), reachability=np.array(reachability), isolation=np.array(isolation), raw_node_feature_list=np.array(raw_node_feature_list), next_hop_indices=np.array(next_hop_indices)), reward, done, {}

    # def step(self, action):
    #     # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    #     # get all necessary information before updating network state
    #     flow_id, link_id = action
    #     done = False
    #     reward = -0.1
    #     link_switches = self.link_switch_map[link_id]
    #     nxt_switch_id = link_switches[0] if link_switches[1] == self.flow_switch_map[flow_id][-1] else link_switches[1]
    #     src_switch_id, dst_switch_id = self.flow_details[flow_id]
    #     connected_components, is_cyclic = self._connected_components(flow_id, visited=False)

    #     # get rewards and check game over conditions
    #     # check if the flow reaches desired destination
    #     if nxt_switch_id == dst_switch_id:
    #         reward = POS_INF
    #         self.incomplete_flows.remove(flow_id)
    #         self.completed_flows.append(flow_id)
    #     else:
    #         # check if the flow reaches wrong edge switch or there is a cycle
    #         if self._is_edge_switch(nxt_switch_id) or nxt_switch_id in self.flow_switch_map[flow_id]:
    #             reward = NEG_INF
    #             self.incomplete_flows.remove(flow_id)
    #             self.completed_flows.append(flow_id)
    #             self.is_game_over = True
    #             print "GAMEself.state = np.zeros((self.num_flows, self.num_links), dtype=np.uint8) OVER"
    #         # check if the flow cannot reach the destination
    #         else:
    #             for group in connected_components:
    #                 if dst_switch_id in group and nxt_switch_id not in group:
    #                     reward = NEG_INF
    #                     self.incomplete_flows.remove(flow_id)
    #                     self.completed_flows.append(flow_id)
    #                     self.is_game_over = True
    #                     print "GAME OVER"
        
    #     # update all network state
    #     # update the switches visited by flow - valid if action is one of next hop link
    #     self.flow_switch_map[flow_id].append(nxt_switch_id)
    #     # if flow_id in self.completed_flows or state[flow_id-1][link_id-1] == 1: # this should not be happening
    #     self.state[flow_id-1][link_id-1] = 1

    #     if len(self.completed_flows) == self.num_flows or self.is_game_over:
    #         done = True
        
    #     return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        return

    def reset(self):
        self._init_flow_details(initialize=False)
        # self.state = np.zeros((self.num_flows, self.num_links), dtype=np.uint8)
        self._init_state()
        topology = self.state["topology"]
        routes = self.state["routes"]
        reachability = self.state["reachability"]
        isolation = self.state["isolation"]
        # next_hop_features = self.state["next_hop_features"]
        raw_node_feature_list = self.state["raw_node_feature_list"]
        next_hop_indices = self.state["next_hop_indices"]
        return dict(topology=np.array(topology), routes=np.array(routes), reachability=np.array(reachability), isolation=np.array(isolation), raw_node_feature_list=np.array(raw_node_feature_list), next_hop_indices=np.array(next_hop_indices))

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
