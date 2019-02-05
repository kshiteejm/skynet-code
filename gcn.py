'''
    GCN Implementation for our purposes.
    Based on 
        https://github.com/tkipf/gcn
'''    

import tensorflow as tf

def construct_gcn_features(curr_flow, adj_mat, flows, visited, isolation):
    curr_visited = visited[curr_flow].copy()
    for i, j in isolation:
        if i == curr_flow:
            for q in visited[j]:
                curr_visited.add(q)
        elif j == curr_flow:
            for q in visited[i]:
                curr_visited.add(q)

def gcn(adj_mat, features=None):
    '''https://stackoverflow.com/questions/36427219/'''
    kernel = tf.Variable()
    bias = tf.Variable()
    if features is None:
        pass

class GCN(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(gcn, self).__init__()
        self.num_inputs = -1
        self.num_outputs = num_outputs

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        self.kernel = tf.Variable("kernel", shape=(self.num_inputs, self.num_outputs))
        self.bias = tf.Variable("bias", shape=(self.num_outputs,))

    def call(self, adj_mat, features=None):
        pass
