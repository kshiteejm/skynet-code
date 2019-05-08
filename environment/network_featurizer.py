import tensorflow as tf


class NetworkFeaturizer(object):

    GRAPH_SCOPE_NAME = 'featurizer'

    def __init__(self,
                 sess,
                 num_nodes = 2,
                 num_raw_features_per_node = 3,
                 node_feature_size = 2,
                 gnn_rounds = 2,
                 hidden_layer_dimens = [32, 16]):
        
        self.sess = sess
        self.num_nodes = num_nodes
        self.num_raw_features_per_node = num_raw_features_per_node
        self.node_feature_size = 16
        self.gnn_rounds = gnn_rounds
        self.hidden_layer_dimens = hidden_layer_dimens

        self.topology_input = tf.placeholder(tf.float32, shape=[None, num_nodes, num_nodes], name="featurizer_input_matrix")
        self.raw_features_input = tf.placeholder(tf.float32, shape=[None, num_nodes, num_raw_features_per_node])

        self.out = self.create_featurizer_network()

    # Creates a graph for computing features of all passed graphs.
    def create_featurizer_network(self):
        with tf.variable_scope(NetworkFeaturizer.GRAPH_SCOPE_NAME, reuse=tf.AUTO_REUSE):
            # Call featurize_one_graph_network for each graph
            output = tf.map_fn(
                        self.featurize_one_graph_network, 
                        (self.topology_input, self.raw_features_input),
                        dtype=tf.float32
                    )
            return output
    
    # Create a graph to process each node
    def featurize_one_graph_network(self, input_tensors):
        topology = input_tensors[0]
        raw_features = input_tensors[1]

        node_idx = tf.constant([i for i in range(self.num_nodes)])
        # Squeeze because split creates some redundant dimensions
        per_node_neighbors = tf.squeeze(
            tf.split(topology, num_or_size_splits = self.num_nodes))
        per_node_features = tf.squeeze(
            tf.split(raw_features, num_or_size_splits = self.num_nodes))

        # Initialize Processed features to 0. This will be updated per iteration
        processed_features = tf.zeros(
            [self.num_nodes, self.node_feature_size], tf.float32)

        for _ in range(self.gnn_rounds):
            # Call featurize_one_node_network per node.

            processed_features = tf.map_fn(
                                    lambda x: self.featurize_one_node_network(x, processed_features),
                                    (node_idx, per_node_neighbors, per_node_features),
                                    dtype=tf.float32
                                )

        return processed_features

    # Process each node
    def featurize_one_node_network(self, input_tensors, all_processed_features):
        # curr_node_idx is used if we want to use the current node's 
        # processed features as well.
        # curr_node_idx = input_tensors[0]
        curr_node_neighbors = input_tensors[1]
        curr_raw_features = input_tensors[2]
        # curr_node_processed_features = tf.gather(all_processed_features, [curr_node_idx])

        # Make a neighbors mask to extract processed features
        # Not comparing to 1 in case the floating point representation messes up
        neighbor_mask = tf.greater(curr_node_neighbors, tf.constant(0.5))

        neighbors_processed_features = tf.boolean_mask(all_processed_features, neighbor_mask)
        agg_neighbors = tf.reduce_sum(neighbors_processed_features, axis=0)
        agg_neighbors = tf.squeeze(agg_neighbors)

        # Coalesce into one big input layer
        input_layer = tf.concat([agg_neighbors, curr_raw_features], axis = 0)
        input_layer = tf.expand_dims(input_layer, 0)

        prev_layer = input_layer
        prev_dimen = self.node_feature_size + self.num_raw_features_per_node

        layer_idx = 1 
        for curr_dimen in self.hidden_layer_dimens:
            # A weird way to create a variable because TF doesn't like tf.Variable
            #  inside map_fns. This also causes the same variable to be used each time.
            # More information here: https://stackoverflow.com/questions/45789822/tensorflow-creating-variables-in-fn-of-tf-map-fn-returns-value-error
            W_hidden = tf.get_variable(
                        ("Featurizer_Hidden_Weight_%d" % (layer_idx)), 
                        shape=[prev_dimen, curr_dimen],
                        dtype=tf.float32,
                        trainable=True,
                        initializer=tf.random_normal_initializer()
                    )
            hidden_layer = tf.nn.relu(tf.matmul(prev_layer, W_hidden), 
                                      name = ("Featurizer_Hidden_%d" % (layer_idx)))
            
            prev_layer = hidden_layer
            layer_idx = layer_idx + 1
            prev_dimen = curr_dimen

        W_out = tf.get_variable(
                "Featurizer_Output_Weight",
                shape=[prev_dimen, self.node_feature_size],
                dtype=tf.float32,
                trainable=True,
                initializer=tf.random_normal_initializer()
            )
        output_layer = tf.nn.relu(tf.matmul(prev_layer, W_out,
                                  name = "Featurizer_Output"))
        output_layer = tf.nn.softmax(output_layer)
        return output_layer

    def calculate_features(self, topology, raw_features):
        # Use the writer to view computed graph
        # self.sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter('logs', self.sess.graph)
        return self.sess.run(self.out, feed_dict = {
            self.topology_input: topology,
            self.raw_features_input: raw_features
            }
        )
        # writer.close()
