from __future__ import print_function

import logging
import time
import threading
import sys

import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug
from keras.layers import Input, Concatenate, Flatten, Dense, Reshape
from keras.models import Model

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
        self.session = tf.Session()

        # self.topology_shape = OBSERVATION_SPACE.spaces["topology"].shape
        # self.routes_shape = OBSERVATION_SPACE.spaces["routes"].shape
        # self.reachability_shape = OBSERVATION_SPACE.spaces["reachability"].shape
        # self.action_shape_height = int(ACTION_SPACE.high[0])
        # self.action_shape_width = int(ACTION_SPACE.high[1])
        
        self.node_feature_size = node_feature_size 
        self.gnn_rounds = gnn_rounds
        self.policy_feature_size = policy_feature_size
        self.feature_size = self.policy_feature_size + self.node_feature_size

        self.net_width = net_width
        # self.next_hop_priority_graph = self._build_next_hop_priority_graph()
        # self.next_hop_policy_graph = self._build_next_hop_policy_graph()

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

    def _build_featurize_graph(self, topology):
        input_node_features = tf.placeholder(tf.float32, shape=(topology.shape[0], self.node_feature_size))
        all_node_features = input_node_features
        for _ in range(self.gnn_rounds):
            for node, edge_list in enumerate(topology):
                ngbr_node_features = tf.boolean_mask(all_node_features[node], edge_list, axis=0)
                summed_ngbr_node_features = tf.reduce_sum(ngbr_node_features)
                node_features = all_node_features[node]
                # Assumption: This will cause the thetas to be reused over multiple rounds, and multiple data points
                with tf.variable_scope("featurize_ngbrs", reuse=True):
                    dense_ngbr_layer = tf.layers.dense(summed_ngbr_node_features, self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_ngbrs")
                    dense_node_layer = tf.layers.dense(node_features, self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_node")
                    all_node_features[node] = tf.reduce_sum([dense_node_layer, dense_ngbr_layer])
        return input_node_features, all_node_features

    def _build_next_hop_priority_graph(self, node_features, policy_features):
        inputs = tf.concat([node_features, policy_features], axis=0)
        with tf.variable_scope("priority_graph"): 
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Input Hops:"), inputs, ":Shape:", tf.shape(inputs))
                with tf.control_dependencies([print_op]):
                    dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            else:
                dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                with tf.variable_scope("dense_priority_1", reuse=True):
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

            return inputs, out_priority

    def _build_next_hop_policy_graph(self, state):

        topology = state["topology"]
        num_flows = state["num_flows"]
        next_hop_indices = state["next_hop_indices"]
        
        raw_node_feat_list = []
        node_feat_list = []
        policy_feat_list = []
        prob_list = []
        reward_list = []
        next_hop_feature_list = []
        
        for flow_id in range(num_flows):
            raw_node_features, node_features = self._build_featurize_graph(topology)
            raw_node_feat_list.append(raw_node_features)
            node_feat_list.append(node_features)

            policy_features = tf.placeholder(tf.float32, shape=self.policy_feature_size)
            policy_feat_list.append(policy_features)

            actual_probabilities = tf.placeholder(tf.float32, shape=(None,))
            prob_list.append(actual_probabilities)

            actual_rewards = tf.placeholder(tf.float32, shape=(None,))
            reward_list.append(actual_rewards)

            per_flow_next_hop_features = tf.gather(node_features, next_hop_indices)
            next_hop_feature_list.append(per_flow_next_hop_features)

        next_hop_features = tf.convert_to_tensor(next_hop_feature_list)
        actual_probabilities = tf.convert_to_tensor(prob_list)
        actual_rewards = tf.convert_to_tensor(reward_list)

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Actual Next Hop Features:Shape:"), tf.shape(next_hop_features))
            with tf.control_dependencies([print_op]):
                avg_next_hop_features = tf.reduce_mean(next_hop_features, axis=1)
        else:
            avg_next_hop_features = tf.reduce_mean(next_hop_features, axis=1)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Average Next Hop Features:"), avg_next_hop_features, ":Shape:", tf.shape(avg_next_hop_features))
            with tf.control_dependencies([print_op]):
                dense_layer = tf.layers.dense(avg_next_hop_features, self.net_width, activation=tf.nn.relu, name="dense_policy_1")
        else:
            dense_layer = tf.layers.dense(avg_next_hop_features, self.net_width, activation=tf.nn.relu, name="dense_policy_1")
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            with tf.variable_scope("dense_policy_1", reuse=True):
                weights = tf.get_variable("kernel")
                print_op = tf.print(Colorize.highlight("Policy Graph: Reward Dense Layer 1 Weights:"), weights, ":Shape:", tf.shape(weights))
            with tf.control_dependencies([print_op]):
                avg_rewards = tf.layers.dense(dense_layer, 1, name="reward") # linear activation
        else:
            avg_rewards = tf.layers.dense(dense_layer, 1, name="reward") # linear activation
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Average Rewards Estimate:"), avg_rewards, ":Shape:", tf.shape(avg_rewards))
            with tf.control_dependencies([print_op]):
                with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                    priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x, policy_features), next_hop_features)
        else:
            with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x, policy_features), next_hop_features)
        
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
        grads_and_vars = self.optimizer.compute_gradients(loss_total)
        minimize = self.optimizer.minimize(loss_total)

        return (raw_node_feat_list, node_feat_list, policy_feat_list, 
                prob_list, reward_list, minimize, next_hop_probabilities, 
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

        # logging.info("ACTIONS: %s", actions)
        # for i in range(0, len(states)):
        #     # logging.info("State: %s", state)
        #     state = states[i]
        #     action = actions[i]
        #     if len(state) == 0:
        #         logging.info("Features Shape: %s, Action Shape: %s", state, action[0].shape)
        #     if len(state) > 0:
        #         logging.info("Features Shape: %s, Action Shape: %s", state[0].shape, action[0].shape)

        # next_hop_features = np.array(states)  
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        # next_hop_features_ = np.array(states_)

        if len(states) > 5*self.min_batch:
            logging.debug("Optimizer alert! Minimizing batch of %d", len(states))

        # actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, grads_and_vars = self.next_hop_policy_graph
        grad = 0.0
        count = 0
        for i in range(0, len(states)):
            if len(states[i]) == 0:
                continue
            
            next_hop_feature = np.vstack([states[i]])
            action = np.vstack([actions[i]])
            reward = np.vstack([rewards[i]])
            next_hop_feature_ = np.vstack([states_[i]])

            logging.debug("Next Hop Feature Shape: %s", next_hop_feature.shape)
            logging.debug("Action: %s, Shape: %s", action, action.shape)
            logging.debug("Reward: %s, Shape: %s", reward, reward.shape)
            logging.debug("Next Hop Feature Last Shape: %s", next_hop_feature_.shape)

            if next_hop_feature_[0].size == 0:
                avg_reward = 0.0
            else:
                avg_reward = self.predict_avg_reward(next_hop_feature_) # self.session.run(avg_rewards_estimate, feed_dict={actual_next_hop_features: next_hop_feature_})
            
            reward = reward + self.gamma_n * avg_reward * np.array([state_masks[i]])

            # with tf.variable_scope("priority_graph"):
            logging.debug("==================START TRAINING=================")
            m, gv = self.session.run([minimize, grads_and_vars], feed_dict={actual_next_hop_features: next_hop_feature, actual_probabilities: action, actual_rewards: reward})
            gv = np.array(gv)
            grad = grad + gv[:, 0]
            count += len(states[i])
            logging.debug("==================END TRAINING=================")
        return grad, count

        # avg_rewards = self.session.run(avg_rewards_estimate, feed_dict={actual_next_hop_features: next_hop_features_})
        # rewards = rewards + self.gamma_n * avg_rewards * state_masks
        # self.session.run(minimize, feed_dict={actual_next_hop_features: next_hop_features, actual_probabilities: actions, actual_rewards: rewards})
        
    def train_push(self, state, action, reward, state_):
        logging.debug("Training Datum: Actual Next Hops Shape: %s, Action: %s, Reward: %s", str(state.shape), str(action), str(reward))

        with self.lock_queue:
            # print "routes shape: %s" % str(state.shape)
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
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self.next_hop_policy_graph
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict={actual_next_hop_features: state})
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict={actual_next_hop_features: state})
            return probabilities, avg_reward

    def predict_prob(self, state):
        with self.default_graph.as_default():
            # print state[0].shape
            # print state[1].shape
            # print state[2].shape
            # np.save('predict_prob_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
            # probabilities, avg_reward = self.model.predict(state)
            logging.debug("PREDICTING PROB.")
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self.next_hop_policy_graph
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict={actual_next_hop_features: state})
            return probabilities

    def predict_avg_reward(self, state):
        with self.default_graph.as_default():
            # np.save('predict_avg_reward_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
            # prob, avg_reward = self.model.predict(state)
            logging.debug("PREDICTING AVG REWARD.")
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate, _ = self.next_hop_policy_graph
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict={actual_next_hop_features: state})
            return avg_reward
