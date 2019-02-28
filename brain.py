from __future__ import print_function

import logging
import time
import threading
import sys
from timeit import default_timer as timer

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
    lock_model = threading.Lock()

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
        self.initialized = False

        self.node_feature_size = node_feature_size 
        self.gnn_rounds = gnn_rounds
        self.policy_feature_size = 3 * policy_feature_size
        self.feature_size = self.policy_feature_size + self.node_feature_size

        self.raw_node_feat_size = 3 * policy_feature_size #ToDo Check up!
        self.net_width = net_width
        
        self.next_hop_policy_graph = self._build_next_hop_policy_graph()
        
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()


    def featurize_node(self, raw_node_feature, neighbor_features): 
        summarized_neighbor_features = tf.reduce_sum(neighbor_features, axis=0)
        with tf.variable_scope("featurize_node", reuse=tf.AUTO_REUSE):
            dense_neighbor_layer = tf.layers.dense(tf.expand_dims(summarized_neighbor_features, 0), 
                self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_neighbors")
            dense_node_layer = tf.layers.dense(tf.expand_dims(raw_node_feature, 0), 
                self.node_feature_size, activation=tf.nn.relu, name="dense_featurize_node")
            dense_neighbor_layer = tf.squeeze(dense_neighbor_layer, axis=[0])
            dense_node_layer = tf.squeeze(dense_node_layer, axis=[0])
            updated_node_feature = tf.reduce_sum([dense_node_layer, dense_neighbor_layer], axis=0)
            return updated_node_feature

    def get_all_node_features(self, index, out_features, raw_node_features, node_features, topology):
        zero = tf.constant(0, dtype=tf.float32)
        neighbor_info = tf.gather(topology, index)
        raw_node_feature = tf.gather(raw_node_features, index)
        neighbor_indices = tf.where(tf.not_equal(neighbor_info, zero))
        neighbor_features = tf.gather(node_features, neighbor_indices)
        new_node_feature = self.featurize_node(raw_node_feature, neighbor_features)
        return [tf.add(index, 1), 
        tf.concat([[new_node_feature], out_features], axis=1), raw_node_features, node_features, topology]

    # featurize for a single flow graph
    def _build_featurize_graph(self, topology, input_node_features):
        # input_node_features = tf.placeholder(tf.float32, shape=(tf.shape(topology)[0], self.raw_node_feature_size))
        all_node_features = tf.zeros((tf.shape(topology)[0], self.node_feature_size))
        for _ in range(self.gnn_rounds):
            node_index = tf.constant(0)
            out_features = tf.Variable([])
            condition = (lambda index, out_features, raw_node_features, node_features, topology: 
                            tf.less(index, tf.shape(topology)[0]))
            _, next_node_features, _, _, _ = tf.while_loop(condition, self.get_all_node_features, 
                [node_index, out_features, input_node_features, all_node_features, topology])
            all_node_features = next_node_features
            # print(all_node_features)
        return all_node_features

    def _build_next_hop_priority_graph(self, inputs):
        with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE): 
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Input Hops:"), inputs, ":Shape:", tf.shape(inputs))
                with tf.control_dependencies([print_op]):
                    dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            else:
                dense_layer = tf.layers.dense(inputs, self.net_width, activation=tf.nn.relu, name="dense_priority_1")
            
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                with tf.variable_scope("dense_priority_1", reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable("kernel")
                    print_op = tf.print(Colorize.highlight("Priority Graph: Priority Dense Layer 1 Weights:"), 
                                weights, ":Shape:", tf.shape(weights))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.layers.dense(dense_layer, 1, name="priority")
            else:
                out_priority = tf.layers.dense(dense_layer, 1, name="priority")

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Next Hop Raw Priorities:"), 
                            out_priority, ":Shape:", tf.shape(out_priority))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.squeeze(out_priority, axis=-1)
            else:
                out_priority = tf.squeeze(out_priority, axis=-1)

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op = tf.print(Colorize.highlight("Priority Graph: Next Hop Priorities:"), out_priority, 
                    ":Shape:", tf.shape(out_priority))
                with tf.control_dependencies([print_op]):
                    out_priority = tf.identity(out_priority)

            return out_priority

    def _build_per_flow_feature_graph(self, flow_id, topology, all_next_hop_indices, all_policy_features, 
            all_raw_node_features, node_feature_list, next_hop_feature_list, priority_feature_list):
        next_hop_indices = tf.gather(all_next_hop_indices, flow_id)
        is_empty = tf.equal(tf.size(next_hop_indices), 0)
        
        raw_node_features = tf.gather(all_raw_node_features, flow_id)
        node_features = self._build_featurize_graph(topology, raw_node_features)
        print_op_nf = tf.print("BRAIN: TF: Raw Node Features:", raw_node_features,
            "BRAIN: TF: Output Node Features: ", node_features)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            with tf.control_dependencies([print_op_nf]):
                next_hop_features = tf.gather(node_features, next_hop_indices)
        else:
            next_hop_features = tf.gather(node_features, next_hop_indices)


        print_op_nhf = tf.print("BRAIN: TF: Per Flow Next Hop Features: ", next_hop_features)
        policy_features = tf.gather(all_policy_features, flow_id)
        priority_features = tf.map_fn(lambda x: tf.concat([x, policy_features], axis=0), next_hop_features)        

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            with tf.control_dependencies([print_op_nhf]):
                priority_features = tf.identity(priority_features)

        print_op = tf.cond(is_empty, true_fn=lambda: tf.print("BRAIN: No Next Hop Features for Flow", flow_id))

        with tf.control_dependencies([print_op]):
            nf = tf.cond(is_empty, true_fn=lambda: node_feature_list, 
                            false_fn=lambda: tf.concat([node_feature_list, tf.expand_dims(node_features, 0)], axis=0))
            nhf = tf.cond(is_empty, true_fn=lambda: next_hop_feature_list, 
                            false_fn=lambda: tf.concat([next_hop_feature_list, tf.expand_dims(next_hop_features, 0)], axis=0))
            prf = tf.cond(is_empty, true_fn=lambda: priority_feature_list, 
                            false_fn=lambda: tf.concat([priority_feature_list, priority_features], axis=0))
        
        return [tf.add(flow_id, 1), topology, all_next_hop_indices, all_policy_features, all_raw_node_features, nf, nhf, prf]
        

    def _build_next_hop_policy_graph(self):
        topology = tf.placeholder(tf.float32, shape=(None, None)) # topology = state["topology"]
        num_flows = tf.placeholder(tf.int32, shape=(1,)) #state["isolation"].shape[0]
        
        all_next_hop_indices = tf.placeholder(tf.int32, shape=(num_flows, None)) #state["next_hop_indices"]
        all_policy_features = tf.placeholder(tf.float32, shape=(num_flows, self.policy_feature_size))
        all_raw_node_features = tf.placeholder(tf.float32, shape=(tf.shape(topology)[0], self.raw_node_feat_size))

        flow_id = tf.constant(0)
        cond = lambda i, topo, anhi, apf, arnf, nfl, nhfl, pfl : tf.less(i, num_flows)
        body = self._build_per_flow_feature_graph
        loop_vars = [flow_id, topology, all_next_hop_indices, all_policy_features, 
                        all_raw_node_features, tf.Variable([]), tf.Variable([]),
                        tf.Variable([])]

        _, _, _, _, _, node_features, next_hop_features, priority_features = \
            tf.while_loop(cond, body, loop_vars)
        
        priority_features = tf.expand_dims(priority_features, 0)
        next_hop_features = tf.expand_dims(next_hop_features, 0)
        logging.debug("BRAIN: TF: All Flows Priority Features: %s", priority_features)
        logging.debug("BRAIN: TF: All Flows Next Hop Features: %s", next_hop_features)
        actual_probabilities = tf.placeholder(tf.float32, shape=(None,None))
        actual_rewards = tf.placeholder(tf.float32, shape=(None,1))
        
        with tf.variable_scope("reward_model", reuse=tf.AUTO_REUSE):
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_op_0 = tf.print(Colorize.highlight("Policy Graph: Raw Node Feature List:"), 
                    all_raw_node_features, ":Shape:", tf.shape(all_raw_node_features))
                print_op_1 = tf.print(Colorize.highlight("Policy Graph: Priority Features:"), 
                    priority_features, ":Shape:", tf.shape(priority_features))
                print_op_2 = tf.print(Colorize.highlight("Policy Graph: Next Hop Features: "), 
                    next_hop_features, ":Shape:", tf.shape(next_hop_features))
                with tf.control_dependencies([print_op_0, print_op_1, print_op_2]):
                    avg_next_hop_features = tf.reduce_mean(next_hop_features, axis=1)
                    # avg_next_hop_features = tf.expand_dims(avg_next_hop_features, 0)
                    logging.debug("BRAIN: TF: All Flows Avg Hop Features: %s", avg_next_hop_features)
                    dense_reward_layer = tf.layers.dense(avg_next_hop_features, self.net_width, 
                        activation=tf.nn.relu, name="dense_policy_1")
                    avg_rewards = tf.layers.dense(dense_reward_layer, 1, name="reward") # linear activation
            else:
                avg_next_hop_features = tf.reduce_mean(next_hop_features, axis=1)
                # avg_next_hop_features = tf.expand_dims(avg_next_hop_features, 0)
                logging.debug("BRAIN: TF: All Flows Avg Hop Features: %s", avg_next_hop_features)
                dense_reward_layer = tf.layers.dense(avg_next_hop_features, self.net_width, 
                    activation=tf.nn.relu, name="dense_policy_1")
                avg_rewards = tf.layers.dense(dense_reward_layer, 1, name="reward") # linear activation
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Average Rewards Estimate:"), avg_rewards, 
                ":Shape:", tf.shape(avg_rewards))
            with tf.control_dependencies([print_op]):
                with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                    priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x), priority_features)
        else:
            with tf.variable_scope("priority_graph", reuse=tf.AUTO_REUSE):
                priorities = tf.map_fn(lambda x: self._build_next_hop_priority_graph(x), priority_features)
        
        priorities = tf.stack(priorities, axis=0)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Priorities Estimate:"), 
                priorities, ":Shape:", tf.shape(priorities))
            with tf.control_dependencies([print_op]):
                next_hop_probabilities = tf.map_fn(tf.nn.softmax, priorities)
        else:
            next_hop_probabilities = tf.map_fn(tf.nn.softmax, priorities)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op = tf.print(Colorize.highlight("Policy Graph: Next Hop Probabilities:"), 
                next_hop_probabilities, ":Shape:", tf.shape(next_hop_probabilities))
            with tf.control_dependencies([print_op]):
                next_hop_probabilities = tf.identity(next_hop_probabilities)
        
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_op_1 = tf.print(Colorize.highlight("Policy Graph: Actual Probabilities:"), 
                actual_probabilities, ":Shape:", tf.shape(actual_probabilities))
            print_op_2 = tf.print(Colorize.highlight("Policy Graph: Actual Rewards:"), actual_rewards, 
                ":Shape:", tf.shape(actual_rewards))
            with tf.control_dependencies([print_op_1, print_op_2]):
                log_prob = tf.log(tf.reduce_sum(next_hop_probabilities * actual_probabilities) + 1e-10)
                advantage = actual_rewards - avg_rewards
        else:
            log_prob = tf.log(tf.reduce_sum(next_hop_probabilities * actual_probabilities) + 1e-10)
            advantage = actual_rewards - avg_rewards

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = self.loss_v * tf.square(advantage)    # minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(next_hop_probabilities * tf.log(next_hop_probabilities + 1e-10), 
            axis=1, keepdims=True) # maximize entropy (regularization)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        with tf.variable_scope("optimize_function", reuse=tf.AUTO_REUSE):
            grads_and_vars = self.optimizer.compute_gradients(loss_total)
            minimize = self.optimizer.minimize(loss_total)

        if not self.initialized:
            self.session.run(tf.global_variables_initializer())
            self.initialized = True
        
        return (topology, num_flows, all_next_hop_indices,
                all_raw_node_features, all_policy_features, 
                actual_probabilities, actual_rewards, minimize, 
                next_hop_probabilities, avg_rewards, grads_and_vars)

    def optimize(self):
        (topology, num_flows, all_next_hop_indices,
            all_raw_node_features, all_policy_features, 
            actual_probabilities, actual_rewards, minimize, 
            next_hop_probabilities,
            avg_rewards, grads_and_vars) = self.next_hop_policy_graph
        if len(self.train_queue[0]) < self.min_batch:
            time.sleep(0)
            return 0.0, 0
        
        with self.lock_queue:
            if len(self.train_queue[0]) < self.min_batch:
                return 0.0, 0
            
            states, actions, rewards, states_, state_masks = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
        
        self.train_iteration = self.train_iteration + 1

        logging.info("OPTIMIZER: =================================== BEGINS ===================================")

        if len(states) > 5*self.min_batch:
            logging.debug("Optimizer alert! Minimizing batch of %d", len(states))

        grad = 0.0
        count = 0
        for i in range(0, len(states)):
            if len(states[i]) == 0:
                continue

            feed_dict = {
                topology: states[i][0]["topology"],
                num_flows: states[i][0]["num_flows"],
                all_next_hop_indices: states[i][0]["next_hop_indices"],
                all_policy_features: Environment.getPolicyFeatures(states[i][0]),
                all_raw_node_features: np.array(states[i][0]["raw_node_feature_list"])
            }
            
            logging.debug("OPTIMIZER: Train Queue Datum Contents: State: %s, Action: %s, Reward: %s", 
                states[i], actions[i], rewards[i])
            raw_node_feature_list = states[i][0]["raw_node_feature_list"]
            action = np.vstack([actions[i]])
            reward = np.vstack([rewards[i]])
            raw_node_feature_list_ = states_[i][0]["raw_node_feature_list"]
            # exit(1)

            logging.debug("OPTIMIZER: Raw Node Feature List Shape: %s", raw_node_feature_list.shape)
            logging.debug("OPTIMIZER: Action: %s, Shape: %s", action, action.shape)
            logging.info("OPTIMIZER: Reward: %s, Shape: %s", reward, reward.shape)
            logging.debug("OPTIMIZER: Raw Node Feature List Last Shape: %s", raw_node_feature_list_.shape)

            if raw_node_feature_list_[0].size == 0:
                avg_reward = 0.0
            else:
                avg_reward = self.predict_avg_reward(states[i][0])
            
            logging.info("OPTIMIZER: Avg Reward: %s, Shape: %s", avg_reward, avg_reward.shape)
            reward = reward + self.gamma_n * avg_reward * np.array([state_masks[i]])
            logging.info("OPTIMIZER: Reward: %s, Shape: %s", reward, reward.shape)
            

            logging.info("==================START TRAINING=================")
            # with self.lock_model:
            with self.default_graph.as_default():
                start_time = timer()
                end_time = timer()
                logging.info("OPTIMIZER: Time to build Training Graph: %s", (end_time - start_time))
                # feed_dict = {i: d for i, d in zip(raw_node_feat_list, raw_node_feature_list)}
                feed_dict[actual_probabilities] = action
                feed_dict[actual_rewards] = reward
                start_time = timer()
                m, gv = self.session.run([minimize, grads_and_vars], feed_dict=feed_dict)
                end_time = timer()
                logging.info("OPTIMIZER: Time to run Training Graph: %s", (end_time - start_time))
                gv = np.array(gv)
                grad = grad + gv[:, 0]
                count += len(states[i])
            logging.info("==================END TRAINING=================")
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
        # with self.default_graph.as_default():
        (topology, num_flows, all_next_hop_indices,
            all_raw_node_features, all_policy_features, 
            _, _, _,next_hop_probabilities_estimate,
            avg_rewards_estimate, _) = self.next_hop_policy_graph

        feed_dict = {
            topology: state["topology"],
            num_flows: state["num_flows"],
            all_next_hop_indices: state["next_hop_indices"],
            all_policy_features: Environment.getPolicyFeatures(state),
            all_raw_node_features: np.array(state["raw_node_feature_list"])
        }

        probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict=feed_dict)
        avg_reward = self.session.run(avg_rewards_estimate, feed_dict=feed_dict)

        return probabilities, avg_reward

    def predict_prob(self, state):
        # with self.lock_model:
        with self.default_graph.as_default():
            logging.debug("BRAIN: Predicting Action Space PDF.")
            # logging.debug("Shape of Raw Node Feature List: %s", state["raw_node_feature_list"].shape)
            # start_time = timer()
            (topology, num_flows, all_next_hop_indices,
                all_raw_node_features, all_policy_features, 
                _, _, _, next_hop_probabilities_estimate, _, _) = self.next_hop_policy_graph
            # end_time = timer()
            # logging.info("BRAIN: Time to Build Action Space PDF Graph: %s", (end_time - start_time))

            feed_dict = {
                topology: state["topology"],
                num_flows: state["num_flows"],
                all_next_hop_indices: state["next_hop_indices"],
                all_policy_features: Environment.getPolicyFeatures(state),
                all_raw_node_features: np.array(state["raw_node_feature_list"])
            }

            start_time = timer()
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict=feed_dict)
            end_time = timer()
            logging.info("BRAIN: Time to Run Action Space PDF Graph: %s", (end_time - start_time))
            return probabilities

    def predict_avg_reward(self, state):
        # with self.lock_model:
        with self.default_graph.as_default():
            logging.debug("BRAIN: Predicting Expected Reward.")
            # logging.debug("Shape of Raw Node Feature List: %s", state["raw_node_feature_list"].shape)
            # start_time = timer()
            (topology, num_flows, all_next_hop_indices,
                all_raw_node_features, all_policy_features, 
                _, _, _, _, avg_rewards_estimate, _) = self.next_hop_policy_graph
            # end_time = timer()
            # logging.info("BRAIN: Time to Build Expected Reward Graph: %s", (end_time - start_time))

            feed_dict = {
                topology: state["topology"],
                num_flows: state["num_flows"],
                all_next_hop_indices: state["next_hop_indices"],
                all_policy_features: Environment.getPolicyFeatures(state),
                all_raw_node_features: np.array(state["raw_node_feature_list"])
            }

            start_time = timer()
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict=feed_dict)
            end_time = timer()
            logging.info("BRAIN: Time to Run Expected Reward Graph: %s", (end_time - start_time))
            return avg_reward
