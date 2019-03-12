import networkx as nx

class NetworkGraph:
    ''' Generates the Network Topology Graph.
        The possible choices are -
        <1> RandomEdgeTopo(n, p): 
        Graph with n nodes and sampling from 
        all possible edges with p probability
        <2> RandomGraphTopo(n, m): 
        Graph with n nodes and m edges 
        randomly sampled
        <3> FatTreeTopo(k): 
        Graph of fat-tree configuration with 
        k pods '''
    
    def __init__(self, type='FatTree', kwargs={"pods": 4}):
        self.graph = self.create_graph(type, kwargs)

    def create_graph(self, type, kwargs):
        graph = None
        if type == 'RandomEdge':
            graph = self.random_edge_topo(**kwargs)
        if type == 'RandomGraph':
            graph = self.random_graph_topo(**kwargs)
        if type == 'FatTree':
            graph = self.fat_tree_topo(**kwargs)
        return graph

    def random_edge_topo(self, nodes, prob):
        return nx.gnp_random_graph(nodes, prob) 

    def random_graph_topo(self, nodes, edges):
        return nx.gnm_random_graph(nodes, edges)
    
    def fat_tree_topo(self, pods=4):
        graph = nx.Graph()

        num_edge_nodes = (pods*pods)//2
        num_agg_nodes = (pods*pods)//2
        num_core_nodes = (pods//2) ** 2
        num_nodes = num_edge_nodes + \
                    num_agg_nodes + \
                    num_core_nodes
        
        graph.add_nodes_from(range(num_nodes))

        for pod in range(pods):
            core_node = 0
            for agg_offset in range(pods//2):
                agg_node = num_core_nodes + \
                           pod*pods + agg_offset
                # core - agg edges
                for _ in range(pods//2):
                    graph.add_edge(core_node, agg_node)
                    core_node += 1
                # agg - edge edges
                for edge_offset in range(pods//2, pods):
                    edge_node = num_core_nodes + \
                                pod*pods + edge_offset
                    graph.add_edge(agg_node, edge_node) 

        return graph
