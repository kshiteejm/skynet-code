import networkx as nx
import numpy as np

class FlowGraph:
    ''' 
    Generates the Flow Graph. 
    Each flow has its own instance 
    of a network graph. 
    This graph contains the flow 
    details and the 
    raw node features. 
    '''
    # raw_edge_feature_names = ["visited", "isolated"]
    raw_node_feature_names = ["visited", "isolated"]

    def __init__(self, graph, src, dst):
        self.graph = self.init_graph(graph, src)
        self.src = src
        self.dst = dst
        self.curr_node = src
        self.done = False
        self.unreachable = False

    ''' 
    Input : Network Graph.
    Output: Network Graph 
            with initialized
            raw features a.k.a.
            Flow Graph.
    '''
    def init_graph(self, graph, src):
        g = graph.copy()
        for n in g.nodes():
            for name in self.raw_node_feature_names:
                g.node[n][name] = 0
        g.node[src]["visited"] = 1
        return g
    
    ''' 
    API's exposed to environment:
    1. step(node): 
       mark next hop node on flow graph.
    2. set_isolated_node(node):
       mark given node as isolated because 
       of isolation contraint in environment.
    3. get_next_hops(): 
       get set of valid next hops.
    4. get_raw_node_features(): 
       get raw node features for all 
       nodes in the flow graph. 
    5. get_topology(): 
       get adjacency matrix for the 
       flow graph. 
    '''
    def step(self, action_node):
        g = self.graph
        dst = self.dst
        g.node[action_node]["visited"] = 1
        self.curr_node = action_node
        if action_node == dst:
            self.done = True
        return self.done
    
    def set_isolated_node(self, isolated_node):
        g = self.graph
        g.node[isolated_node]["isolated"] = 1

    def get_next_hops(self):
        g = self.graph
        curr_node = self.curr_node
        dst = self.dst
        reachable_to_dst_next_hops = []

        can_visit_nodes = [
            n for n, d in g.nodes(data=True) 
            if d["visited"] == 0 and d["isolated"] == 0
        ]
        sg = g.subgraph(can_visit_nodes)
        
        next_hop_iterator = g.neighbors(curr_node)
        for next_hop in next_hop_iterator:
            if next_hop in sg:
                if nx.has_path(sg, next_hop, dst):
                    reachable_to_dst_next_hops.append(next_hop)
        
        if len(reachable_to_dst_next_hops) == 0:
            self.unreachable = True
            self.done = True
        
        return self.unreachable, reachable_to_dst_next_hops
    
    def get_raw_node_features(self):
        g = self.graph
        raw_node_feature_names = self.raw_node_feature_names
        num_nodes = g.number_of_nodes()
        raw_node_feature_size = len(raw_node_feature_names)
        raw_node_features = np.zeros((num_nodes, raw_node_feature_size))
        for n in g.nodes():
            for i in range(len(raw_node_feature_names)):
                raw_node_features[n][i] = g.node[n][raw_node_feature_names[i]]
        return raw_node_features
    
    def get_topology(self):
        g = self.graph
        return nx.to_numpy_array(g)

    def is_done(self):
        return self.done
    
    def reset(self):
        self.__init__(self.graph, self.src, self.dst)