class SkynetEnv:
    def __init__(self):
        self.adjacency_matrix = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1]]
        self.adjacency_list = [[1, 2], [0, 3], [0, 4], [1, 4]]
        self.flows = [[0, 4]] 
        self.state = []

        self.switch_s = [0, 1, 2, 3, 
        self.num_host_ids = 2
        self.num_flow_ids = 1
        self.edges = []
        for i in range(0, self.num_switch):
            self.state = [
