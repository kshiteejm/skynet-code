import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as tl

from constants import GAMMA, ENTROPY_WEIGHT, ENTROPY_EPS, S_INFO, LEARNING_RATE

class ActorNetwork(object):
    """
    Input: State i.e. Current Flow Graph
    Output: PDF on Actions i.e. Next Hops for current Flow
    """
    def __init__(self, sess, network_featurizer,
                 state_dim = [1, 1], action_dim = [1, 1], 
                 learning_rate = LEARNING_RATE,
                 hidden_layer_dimens = [32, 16]):
        self.sess = sess
        self.network_featurizer = network_featurizer
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.hidden_layer_dimens = hidden_layer_dimens

        # Create the actor network
        # output of the featurizer becomes input for actor
        self.inputs = self.network_featurizer.out
        self.out = self.create_actor_network()

        # Get all network parameters
        # Ensure that only the actor variables are covered by this
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(
                        tf.multiply(
                            tf.log(
                                tf.reduce_sum(tf.multiply(self.out, self.acts),
                                reduction_indices=1,
                                keep_dims=True)
                            ), 
                            -self.act_grad_weights
                        )
                    ) + ENTROPY_WEIGHT * \
                        tf.reduce_sum(tf.multiply(
                                        self.out, 
                                        tf.log(self.out + ENTROPY_EPS)
                                    ))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        # Optimize only optimizes the Actor graph. 
        # Featurizer sits in a different scope
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        # Now that I look at it, we are only using the processed features as input,
        # I missed adding next hops as another input. 
        with tf.variable_scope('actor') :
            # brain.py network goes here
            
            # Creating the hidden layers. 
            # the dimensions here would have to be changed
            prev_layer = self.inputs
            prev_dimen = tf.shape(self.inputs)#[0]?
            
            # Number of hidden layers can be controlled by specifying 
            # layer dimens in hidden_layer_dimens
            layer_number = 1 # used for naming
            for curr_dimen in self.hidden_layer_dimens:
                w_mat = tf.Variable(tf.random_normal([prev_dimen, curr_dimen]))
                curr_layer = tf.nn.relu(tf.matmul(prev_layer, w_mat), 
                                        name=("Actor_Hidden_%d" % (layer_number))
                             )

                prev_layer = curr_layer
                prev_dimen = curr_dimen
                layer_number = layer_number + 1

            w_out = tf.Variable(tf.random_normal([prev_dimen, self.a_dim]) )
            out_layer = tf.nn.relu(tf.matmul(prev_layer, w_out), name="Actor_Raw_Output")
            
            output_probabilities = tf.nn.softmax(out_layer, name="Actor_Output_Softmaxed")
            
            # Output as one-hot vector of selected action. Uncomment if needed
            # output = tf.one_hot([tf.argmax(output)], self.a_dim)
            return output_probabilities


    def train(self, inputs, acts, act_grad_weights):
        self.sess.run(self.optimize, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"],
            # Will have to add a separate next_hops input
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"]
            # Will have to add a separate next_hops input
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"],
            # Will have to add a separate next_hops input
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        # Only applies gradient to the actor network
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, network_featurizer,
                 state_dim = [1, 1], 
                 learning_rate = LEARNING_RATE,
                 hidden_layer_dimens = [32, 16]):
        self.sess = sess
        self.network_featurizer = network_featurizer
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.hidden_layer_dimens = hidden_layer_dimens

        # Create the critic network
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.out = self.create_critic_network()

        # Get all network parameters
        # We want the critic graph, as well as the featurizer graph. 
        # Simply append the two into a list
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_featurizer.GRAPH_SCOPE_NAME)

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tf.reduce_mean(tf.square(self.td))

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        # This should apply gradients to the critic, and the featurizer
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            # brain.py call here
            prev_layer = self.inputs
            prev_dimen = tf.shape(self.inputs)#[0]?

            layer_number = 1
            for curr_dimen in self.hidden_layer_dimens:
                w_mat = tf.Variable(tf.random_normal([prev_dimen, curr_dimen]))
                # Leaky RelU used because RelU would get rid of all negative values. 
                # Reward belongs to [-Inf, Inf], so activations should be in the same range.
                curr_layer = tf.nn.leaky_relu(tf.matmul(prev_layer, w_mat), 
                                        name=("Critic_Hidden_%d" % (layer_number)))
                
                prev_layer = curr_layer
                prev_dimen = curr_dimen
                layer_number = layer_number + 1

            w_out = tf.Variable(tf.random_normal([prev_dimen, 1]))
            out_layer = tf.nn.leaky_relu(tf.matmul(prev_layer, w_out), name = "Critic_Output")

            return out_layer

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"],
            # Will have to add a separate next_hops input
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"]
            # Will have to add a separate next_hops input
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"],
            # Will have to add a separate next_hops input
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.network_featurizer.topology_input: inputs["topology"],
            self.network_featurizer.raw_features_input: inputs["raw_node_features"],
            # Will have to add a separate next_hops input
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(state_batch, action_batch, reward_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert state_batch.shape[0] == action_batch.shape[0]
    assert state_batch.shape[0] == reward_batch.shape[0]
    ba_size = state_batch.shape[0]

    v_batch = critic.predict(state_batch)

    R_batch = np.zeros(reward_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = reward_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    # this entirely isolates actor and critic graphs
    # Actor and critic independently act on the states.
    actor_gradients = actor.get_gradients(state_batch, action_batch, td_batch)
    critic_gradients = critic.get_gradients(state_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)

    summary_vars = [td_loss, eps_total_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
