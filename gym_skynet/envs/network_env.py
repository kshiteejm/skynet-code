from network_graph import NetworkGraph
from flow_graph import FlowGraph
import itertools

class NetworkEnv():
    NEG_INF = -100

    def __init__(self, topo_type="FatTree", kwargs={"pods": 4}, 
                 num_flows=10, num_node_isolations=2):
        self.network_graph = NetworkGraph(topo_type, kwargs)
        self.flow_graphs, self.flows = self.init_flow_graphs(num_flows)
        self.node_isolations = self.network_graph.get_node_isolations(self.flows, num_node_isolations)
        self.round_robin = itertools.cycle(self.flow_graphs)
        self.flow_id_current = next(self.round_robin)
        self.done = False
        self.unreachable = False
    
    def init_flow_graphs(self, num_flows):
        flow_graphs = {}
        flows = self.network_graph.get_flows(num_flows)
        for flow_id, (src, dst) in flows.items():
            flow_graphs[flow_id] = FlowGraph(self.network_graph, src, dst)
        return flow_graphs, flows

    ''' 
    Reset is called to solve exactly the same 
    problem instance once again. 
    '''
    def reset(self):
        for flow_graph in self.flow_graphs.values():
            flow_graph.reset()
        self.round_robin = itertools.cycle(self.flow_graphs)
        self.flow_id_current = next(self.round_robin)
    
    def get_current_flow_graph(self):
        return self.flow_graphs[self.flow_id_current]
    
    def get_next_flow_graph(self):
        flow_id_begin = next(self.round_robin)
        flow_id_chosen = flow_id_begin
        flow_graph_next = self.flow_graphs[flow_id_chosen]
        while flow_graph_next.is_done():
            flow_id_chosen = next(self.round_robin)
            flow_graph_next = self.flow_graphs[flow_id_chosen]
            if flow_id_chosen == flow_id_begin:
                flow_graph_next = None
                break
        self.flow_id_current = flow_id_chosen 
        return flow_graph_next

    def set_isolated_nodes(self, isolated_node):
        for flow_id in self.node_isolations[self.flow_id_current]:
            flow_graph = self.flow_graphs[flow_id]
            flow_graph.set_isolated_node(isolated_node)

    def step(self, action):
        reward = -1
        flow_graph_current = self.get_current_flow_graph()
        flow_graph_current.step(action)
        self.set_isolated_nodes(action)

        flow_graph_next = self.get_next_flow_graph()
        if flow_graph_next == None:
            self.done = True
        unreachable, next_hops = flow_graph_next.get_next_hops()
        if unreachable:
            reward = self.NEG_INF
        state = dict(
            topology=flow_graph_next.get_topology(),
            raw_node_features=flow_graph_next.get_raw_node_features(),
            next_hops=next_hops
        )

        return state, reward, self.done