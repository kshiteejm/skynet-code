from __future__ import print_function

import logging
import time
import threading
import sys

import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

from environment import Environment
from constants import GAMMA, LEARNING_RATE, N_STEP_RETURN, MIN_BATCH, LOSS_V, LOSS_ENTROPY, \
                    NODE_FEATURE_SIZE, GNN_ROUNDS, POLICY_FEATURE_SIZE, NET_WIDTH, Colorize

class Brain:
    # train_queue = [ [[], [], []], [], [], [[], [], []], [] ]    # s, a, r, s', s' terminal mask
    train_queue = [ [], [], [], [], [] ]
    lock_queue = threading.Lock()

    def __init__(self, gamma=GAMMA, n_step_return=N_STEP_RETURN, 
                learning_rate=LEARNING_RATE, min_batch=MIN_BATCH, loss_v=LOSS_V, 
                loss_entropy=LOSS_ENTROPY, node_feature_size=NODE_FEATURE_SIZE,
                gnn_rounds=GNN_ROUNDS, policy_feature_size=POLICY_FEATURE_SIZE,
                net_width=NET_WIDTH):
        self.train_iteration = 0

        self.gamma = gamma
        self.n_step_return = n_step_return
        self.gamma_n = self.gamma ** self.n_step_return

        self.learning_rate = learning_rate
        self.min_batch = min_batch

        self.loss_v = loss_v
        self.loss_entropy = loss_entropy

        self.optimizer = None

        #Enabling Eager Exec
        # tf.enable_eager_execution()
        self.session = tf.Session()

        self.node_feature_size = node_feature_size 
        self.gnn_rounds = gnn_rounds
        self.policy_feature_size = 3 * policy_feature_size
        self.feature_size = self.policy_feature_size + self.node_feature_size

        self.net_width = net_width
        
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
    
    # featurize for a single flow graph
    def _build_featurize_graph(self, topology, raw_node_feature_size):
        input_node_features = tf.placeholder(tf.float32, shape=(topology.shape[0], raw_node_feature_size))
        all_node_features = tf.zeros((topology.shape[0], self.node_feature_size))
        for _ in range(self.gnn_rounds):
            next_node_features = []
            for node, edge_list in enumerate(topology):
                ngbr_node_features = tf.boolean_mask(all_node_features, edge_list, axis=0)
                summed_ngbr_node_features = tf.reduce_sum(ngbr_node_features, axis=0)
                node_features = input_node_features[node]
                # Assumption: This will cause the thetas to be reused over multiple rounds, and multiple data points
                with tf.variable_scope("featurize_ngbrs", reuse=tf.AUTO_REUSE):
                    dense_ngbr_layer = tf.layers.dense(tf.expand_dims(summed_ngbr_node_features, 0), self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_ngbrs")
                    dense_node_layer = tf.layers.dense(tf.expand_dims(node_features, 0), self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_node")
                    dense_ngbr_layer = tf.squeeze(dense_ngbr_layer, axis=[0])
                    dense_node_layer = tf.squeeze(dense_node_layer, axis=[0])
                    updated_node_feature = tf.reduce_sum([dense_node_layer, dense_ngbr_layer], axis=0)
                    next_node_features.append(updated_node_feature)
            all_node_features = tf.convert_to_tensor(next_node_features)
            # print(all_node_features)
        return input_node_features, all_node_features

    def _build_next_hop_priority_graph(self, inputs):
        inputs = tf.expand_dims(inputs, 0)

        with tf.variable_scope("priority_graph"): 
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Input Hops:"), inputs, ":Shape:", tf.shape(inputs))
                with tf.control_dependencies([print_op]):
                    dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            else:
                dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                with tf.variable_scope("dense_priority_1", reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable("kernel")
                    print_op = tf.print(Colorize.highlight("Priority Graph: Priority Dense Layer 1 Weights:"), weights, ":Shape:", tf.shape(weights))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.layers.dense(dense_layer, 1, name="priority")
            else:
                out_priority = tf.layers.dense(dense_layer, 1, name="priority")

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Next Hop Raw Priorities:"), out_priority, ":Shape:", tf.shape(out_priority))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.squeeze(out_priority, axis=-1)
            else:
                out_priority = tf.squeeze(out_priority, axis=-1)

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Next Hop Priorities:"), out_priority, ":Shape:", tf.shape(out_priority))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.identity(out_priority)

            return out_priority

    def _build_next_hop_policy_graph(self, state):
        topology = state["topology"]
        num_flows = state["isolation"].shape[0]
        next_hop_indices = state["next_hop_indices"]
        raw_node_feature_size = state["raw_node_feature_list"].shape[-1]
        logging.debug("BRAIN: INPUT: Shapes: topology: %s, next_hop_indices: %s, raw_node_feature_list: %s", topology.shape, next_hop_indices.shape, state["raw_node_feature_list"].shape)
        
        raw_node_feat_list = []
        node_feat_list = []
        next_hop_feature_list = []
        priority_feature_list = []

        for flow_id in range(num_flows):
            raw_node_features, node_features = self._build_featurize_graph(topology, raw_node_feature_size)
            raw_node_feat_list.append(raw_node_features)
            node_feat_list.append(node_features)
            logging.debug("BRAIN: TF: Raw Node Features: %s", raw_node_features)
            logging.debug("BRAIN: TF: Output Node Features: %s", node_features)

            per_flow_next_hop_features = tf.gather(node_features, next_hop_indices[flow_id])
            logging.debug("BRAIN: TF: Per Flow Next Hop Features: %s", per_flow_next_hop_features)
            next_hop_feature_list.append(per_flow_next_hop_features)

            policy_features = Environment.getPolicyFeatures(state, flow_id)
            policy_features = tf.convert_to_tensor(policy_features, dtype=tf.float32)
            priority_features = tf.map_fn(lambda x: tf.concat([x, policy_features], axis=0), per_flow_next_hop_features)
            logging.debug("BRAIN: TF: Per Flow Priority Features: %s", priority_features)
            priority_feature_list.append(priority_features)

        priority_features = tf.concat(priority_feature_list, axis=0)
        next_hop_features = tf.concat(next_hop_feature_list, axis=0)
        # priority_features = tf.expand_dims(priority_features, 0)
        # next_hop_features = tf.expand_dims(next_hop_features, 0)
        logging.debug("BRAIN: TF: All Flows Priority Features: %s", priority_features)
        logging.debug("BRAIN: TF: All Flows Next Hop Features: %s", next_hop_features)
        actual_probabilities = tf.placeholder(tf.float32, shape=(None,))
        actual_rewards = tf.placeholder(tf.float32, shape=(None,))
        
        with tf.variable_scope("reward_model", reuse=tf.AUTO_REUSE):
            avg_next_hop_features = tf.reduce_mean(next_hop_features, axis=0)
            avg_next_hop_features = tf.expand_dims(avg_next_hop_features, 0)
            logging.debug("BRAIN: TF: All Flows Avg Hop Features: %s", avg_next_hop_features)
            dense_reward_layer = tf.layers.dense(avg_next_hop_features, self.net_width, activation=tf.nn.relu, name="dense_policy_1")
            avg_rewards = tf.layers.dense(dense_reward_layer, 1, name="reward") # linear activation
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Average Rewards Estimate:"), avg_rewards, ":Shape:", tf.shape(avg_rewards))
            with tf.control_dependencies([print_op]):
                with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                    priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x), priority_features)
        else:
            with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x), priority_features)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Priorities Estimate:"), priorities, ":Shape:", tf.shape(priorities))
            with tf.control_dependencies([print_op]):
                next_hop_probabilities = tf.map_fn(tf.nn.softmax, priorities)
        else:
            next_hop_probabilities = tf.map_fn(tf.nn.softmax, priorities)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Next Hop Probabilities:"), next_hop_probabilities, ":Shape:", tf.shape(next_hop_probabilities))
            with tf.control_dependencies([print_op]):
                next_hop_probabilities = tf.identity(next_hop_probabilities)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op_1 = tf.print(Colorize.highlight("Policy Graph: Actual Probabilities:"), actual_probabilities, ":Shape:", tf.shape(actual_probabilities))
            print_op_2 = tf.print(Colorize.highlight("Policy Graph: Actual Rewards:"), actual_rewards, ":Shape:", tf.shape(actual_rewards))
            with tf.control_dependencies([print_op_1, print_op_2]):
                log_prob = tf.log(tf.reduce_sum(next_hop_probabilities * actual_probabilities) + 1e-10)
                advantage = actual_rewards - avg_rewards
        else:
            log_prob = tf.log(tf.reduce_sum(next_hop_probabilities * actual_probabilities) + 1e-10)
            advantage = actual_rewards - avg_rewards

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = self.loss_v * tf.square(advantage)    # minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(next_hop_probabilities * tf.log(next_hop_probabilities + 1e-10), axis=1, keepdims=True) # maximize entropy (regularization)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        with tf.variable_scope("optimize_function", reuse=tf.AUTO_REUSE):
            grads_and_vars = self.optimizer.compute_gradients(loss_total)
            minimize = self.optimizer.minimize(loss_total)

        return (raw_node_feat_list, actual_probabilities, 
                actual_rewards, minimize, next_hop_probabilities, 
                avg_rewards, grads_and_vars)

    def optimize(self):

        if len(self.train_queue[0]) < self.min_batch:
            time.sleep(0)
            return 0.0, 0
        
        with self.lock_queue:
            if len(self.train_queue[0]) < self.min_batch:
                return 0.0, 0
            
            states, actions, rewards, states_, state_masks = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
        
        self.train_iteration = self.train_iteration + 1

        if len(states) > 5*self.min_batch:
            logging.debug("Optimizer alert! Minimizing batch of %d", len(states))

        grad = 0.0
        count = 0
        for i in range(0, len(states)):
            if len(states[i]) == 0:
                continue
            
            raw_node_feature_list = np.vstack([states[i]["raw_node_feature_list"]])
            action = np.vstack([actions[i]])
            reward = np.vstack([rewards[i]])
            raw_node_feature_list_ = np.vstack([states_[i]["raw_node_feature_list"]])

            logging.debug("Raw Node Feature List Shape: %s", raw_node_feature_list.shape)
            logging.debug("Action: %s, Shape: %s", action, action.shape)
            logging.debug("Reward: %s, Shape: %s", reward, reward.shape)
            logging.debug("Raw Node Feature List Last Shape: %s", raw_node_feature_list_.shape)

            if raw_node_feature_list_[0].size == 0:
                avg_reward = 0.0
            else:
                avg_reward = self.predict_avg_reward(states[i])
            
            reward = reward + self.gamma_n * avg_reward * np.array([state_masks[i]])

            logging.debug("==================START TRAINING=================")
            raw_node_feat_list, actual_probabilities, actual_rewards, minimize, next_hop_probabilities, avg_rewards, grads_and_vars = self._build_next_hop_policy_graph(states[i])
            feed_dict = {i: d for i, d in zip(raw_node_feat_list, raw_node_feature_list)}
            feed_dict[actual_probabilities] = action
            feed_dict[actual_rewards] = reward
            m, gv = self.session.run([minimize, grads_and_vars], feed_dict=feed_dict)
            gv = np.array(gv)
            grad = grad + gv[:, 0]
            count += len(states[i])
            logging.debug("==================END TRAINING=================")
        return grad, count

    def train_push(self, state, action, reward, state_):
        logging.debug("Training Datum: State: %s, Action: %s, Reward: %s", str(state), str(action), str(reward))

        with self.lock_queue:
            self.train_queue[0].append([state])
            self.train_queue[1].append([action])
            self.train_queue[2].append([reward])

            if state_ is None:
                self.train_queue[3].append([state_])
                self.train_queue[4].append([0.])
            else:
                self.train_queue[3].append([state_])
                self.train_queue[4].append([1.])

    def predict(self, state):
        with self.default_graph.as_default():
            raw_node_feature_list, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self._build_next_hop_policy_graph(state)
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict={i: d for i, d in zip(raw_node_feature_list, state["raw_node_feature_list"])})
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict={i: d for i, d in zip(raw_node_feature_list, state["raw_node_feature_list"])})
            return probabilities, avg_reward

    def predict_prob(self, state):
        with self.default_graph.as_default():
            logging.debug("PREDICTING PROB.")
            logging.debug("Shape of Raw Node Feature List: %s", state["raw_node_feature_list"].shape)
            raw_node_feature_list, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self._build_next_hop_policy_graph(state)
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict={i: d for i, d in zip(raw_node_feature_list, state["raw_node_feature_list"])})
            return probabilities

    def predict_avg_reward(self, state):
        with self.default_graph.as_default():
            logging.debug("PREDICTING AVG REWARD.")
            raw_node_feature_list, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self._build_next_hop_policy_graph(state)
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict={i: d for i, d in zip(raw_node_feature_list, state["raw_node_feature_list"])})
            return avg_reward
