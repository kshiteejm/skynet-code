import time
import itertools
import random

import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import gym

import gym_skynet
import deepwalk
from constants import *

class Brain:
    train_queue = [ [[], [], []], [], [], [[], [], []], [] ]    # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, gamma=GAMMA, n_step_return=N_STEP_RETURN, learning_rate=LEARNING_RATE, min_batch=MIN_BATCH, 
                loss_v=LOSS_V, loss_entropy=LOSS_ENTROPY):
        self.train_iteration = 0

        self.gamma = gamma
        self.n_step_return = n_step_return
        self.gamma_n = self.gamma ** self.n_step_return

        self.learning_rate = learning_rate
        self.min_batch = min_batch

        self.loss_v = loss_v
        self.loss_entropy = loss_entropy

        self.session = tf.Session()

        self.topology_shape = OBSERVATION_SPACE.spaces["topology"].shape
        self.routes_shape = OBSERVATION_SPACE.spaces["routes"].shape
        self.reachability_shape = OBSERVATION_SPACE.spaces["reachability"].shape
        self.action_shape_height = int(ACTION_SPACE.high[0])
        self.action_shape_width = int(ACTION_SPACE.high[1])
        self.next_hop_feature_shape = list([node_features[0].shape[0]*4])

        self.next_hop_priority_graph = self._build_next_hop_priority_graph()
        self.next_hop_policy_graph = self._build_next_hop_policy_graph()

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

    def _build_model(self):
        if TOPO:
            topology_input = Input(shape=self.topology_shape)
        routes_input = Input(shape=self.routes_shape)
        reachability_input = Input(shape=self.reachability_shape)

        if TOPO:
            merged_input = Concatenate(axis=1)([topology_input, routes_input, reachability_input])
        else:
            merged_input = Concatenate(axis=1)([routes_input, reachability_input])
        flattened_input = Flatten()(merged_input)
        dense_layer_1 = Dense(256, activation='relu')(flattened_input)
        dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)

        _out_actions = Dense(self.action_shape_height*self.action_shape_width, activation='softmax')(dense_layer_2)
        out_actions = Reshape((self.action_shape_height, self.action_shape_width))(_out_actions)
        out_value = Dense(1, activation='linear')(dense_layer_2)

        if TOPO:
            model = Model(inputs=[topology_input, routes_input, reachability_input], outputs=[out_actions, out_value])
        else:
            model = Model(inputs=[routes_input, reachability_input], outputs=[out_actions, out_value])

        model._make_predict_function()  # have to initialize before threading
        
        if DEBUG:
            print(model.summary())

        return model
    
    def _build_next_hop_priority_graph(self):
        with tf.Graph().as_default() as next_hop_priority_graph:
            next_hop_feature_shape = list(self.next_hop_feature_shape)
            next_hop_feature_shape.insert(0, None)
            next_hop_feature = tf.placeholder(tf.float32, shape=next_hop_feature_shape, name="feature")
            dense_layer = tf.layers.dense(next_hop_feature, 16, activation=tf.nn.relu)
            out_priority = tf.layers.dense(dense_layer, 1, name="priority") # linear activation
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        return next_hop_priority_graph
    
    def _build_nhp_graph(self, inputs):
        with tf.variable_scope("test") as scope: 
            dense_layer = tf.layers.dense(inputs, 16, activation=tf.nn.relu)
            out_priority = tf.layers.dense(dense_layer, 1, name="priority") # linear activation
            return out_priority

    def _build_next_hop_policy_graph(self):
        actual_next_hop_features_shape = list(self.next_hop_feature_shape)
        actual_next_hop_features_shape.insert(0, None)
        actual_next_hop_features_shape.insert(0, None)
        actual_next_hop_features = tf.placeholder(tf.float32, shape=actual_next_hop_features_shape)
        actual_probabilities = tf.placeholder(tf.float32, shape=(None,None,1))
        actual_rewards = tf.placeholder(tf.float32, shape=(None,1))
        
        avg_next_hop_features = tf.reduce_mean(actual_next_hop_features, axis=1)
        dense_layer = tf.layers.dense(avg_next_hop_features, 16, activation=tf.nn.relu)
        avg_rewards = tf.layers.dense(dense_layer, 1, name="reward") # linear activation
        with tf.variable_scope("test", reuse=tf.AUTO_REUSE) as scope:
            priorities = tf.map_fn(lambda x: self._build_nhp_graph(x), actual_next_hop_features)
        next_hop_probabilities = tf.nn.softmax(priorities)
        
        log_prob = tf.log(tf.reduce_sum(next_hop_probabilities * actual_probabilities) + 1e-10)
        advantage = actual_rewards - avg_rewards

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = self.loss_v * tf.square(advantage)    # minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(next_hop_probabilities * tf.log(next_hop_probabilities + 1e-10), axis=1, keepdims=True) # maximize entropy (regularization)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities, avg_rewards

    def optimize(self):
        if len(self.train_queue[0]) < self.min_batch:
            time.sleep(0)
            return
        
        with self.lock_queue:
            if len(self.train_queue[0]) < self.min_batch:
                return
            
            states, actions, rewards, states_, state_masks = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
        
        self.train_iteration = self.train_iteration + 1

        next_hop_features = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_hop_features_ = np.vstack(states_)

        if len(next_hop_features) > 5*self.min_batch:
            print("Optimizer alert! Minimizing batch of %d" % len(next_hop_features))
        
        actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate = self.next_hop_policy_graph

        avg_rewards = self.session.run(avg_rewards_estimate, feed_dict={avg_rewards_estimate})
        rewards = rewards + self.gamma_n * avg_rewards * state_masks

        self.session.run(minimize, feed_dict={actual_next_hop_features: next_hop_features, actual_probabilities: actions, actual_rewards: rewards})

    def train_push(self, state, action, reward, state_):
        if DEBUG:
            print("Training Datum: Actual Next Hops: %s, Action: %s, Reward: %s" % (str(state), str(action), str(reward)))

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
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate = self.next_hop_policy_graph
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
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate = self.next_hop_policy_graph
            probabilities = self.session.run(next_hop_probabilities_estimate, feed_dict={actual_next_hop_features: state})
            return probabilities

    def predict_avg_reward(self, state):
        with self.default_graph.as_default():
            # np.save('predict_avg_reward_%d' % self.train_iteration, [l.get_weights() for l in self.model.layers])
            # prob, avg_reward = self.model.predict(state)
            actual_next_hop_features, actual_probabilities, actual_rewards, minimize, next_hop_probabilities_estimate, avg_rewards_estimate = self.next_hop_policy_graph
            avg_reward = self.session.run(avg_rewards_estimate, feed_dict={actual_next_hop_features: state})
            return avg_reward
