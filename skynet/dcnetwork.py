import numpy as np
import random

NOFLOW = 0
FLOWO = 1

class DCNetworkState(object):
    """State of the datacenter network and some basic functions to interact with it
    """

    def __init__(self, layers=3, topo="tiered", size=2, flows=1):
        if topo == "tiered":
            # number of edge, aggregate, core, and total number of switches
            # switch id 1 to size*size
            self.num_edge_switches = size*size
            # switch id size*size + 1 to size*size + size
            self.num_agg_switches = size
            # switch id size*size + size + 1 to size*size + size + size/2
            self.num_core_switches = size/2
            self.num_switches = self.num_edge_switches + self.num_agg_switches + self.num_core_switches
            # number of network links - edge connects to one agg, and agg connects to all core
            self.num_links = size*size + size*size/2
            # link id to switch ids mapping - link id and switch id contiguous starting from 1
            self.link_to_switches = {}
            for i in range(
            # adjacent switches for each switch, indexed by switch number
            self.connected_switches = []
            # adjacent edges for each switch, indexed by switch number
            self.connected_edges = []
            
            self.num_flows = flows 
            self.flow_details = []
            for i in range(0, num_flows):
                src = random.randint(1, size*size)
                dst = src
                while dst != src:
                    dst = random.randint (1, size*size)
                self.flow_details.append((src, dst))
            self.links = np.zeros((num_links, num_flows), dtype=int)
            self.links.fill(NOFLOW)

