'''
    GCN Implementation for our purposes.
    Based on 
    https://stackoverflow.com/questions/36427219/can-i-write-a-custom-layer-in-tensorflow-in-python-with-existing-ops-such-as-con
    and
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
    kernel = tf.Variable()
    bias = tf.Variable()
    if features is None:
