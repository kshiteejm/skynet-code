from __future__ import print_function # python2 artifact

import logging
import os
import sys
import itertools

import argparse

import numpy as np

import tensorflow as tf
import gym
import gym_skynet

from brain import Brain
from environment import Environment
from optimizer import Optimizer

from constants import EPS_END, EPS_START, EPS_STEPS, \
                    GAMMA, LEARNING_RATE, LOSS_ENTROPY, LOSS_V, MIN_BATCH, \
                    N_STEP_RETURN, GNN_ROUNDS, POLICY_FEATURE_SIZE, \
                    NODE_FEATURE_SIZE, NET_WIDTH, \
                    OPTIMIZERS, PER_INSTANCE_LIMIT, TESTING, \
                    TESTING_INSTANCE_LIMIT, THREAD_DELAY, THREADS, \
                    TRAINING_INSTANCE_LIMIT, \
                    MIN_GRAPH_SIZE, MAX_GRAPH_SIZE, MIN_FLOWS, MAX_FLOWS, \
                    GRAD_NORM_STOP, TOPOLOGIES

brain = None

def main(gamma=GAMMA, n_step_return=N_STEP_RETURN, learning_rate=LEARNING_RATE, 
            min_batch=MIN_BATCH, loss_v=LOSS_V, loss_entropy=LOSS_ENTROPY,
            gnn_rounds=GNN_ROUNDS, node_feature_size=NODE_FEATURE_SIZE,
            policy_feature_size=POLICY_FEATURE_SIZE, net_width=NET_WIDTH,
            eps_start=EPS_START, eps_end=EPS_END, eps_steps=EPS_STEPS, 
            thread_delay=THREAD_DELAY,
            threads=THREADS, optimizers=OPTIMIZERS,
            per_instance_limit=PER_INSTANCE_LIMIT,
            training_instance_limit=TRAINING_INSTANCE_LIMIT,                
            testing_instance_limit=TESTING_INSTANCE_LIMIT,
            test=TESTING):

    global brain
    # TODO fix    

    delete()

    Environment.INSTANCE_NUM = itertools.count()

    brain = Brain(gamma=gamma, n_step_return=n_step_return, 
                learning_rate=learning_rate, min_batch=min_batch, loss_v=loss_v, 
                loss_entropy=loss_entropy, node_feature_size=node_feature_size,
                gnn_rounds=gnn_rounds, policy_feature_size=policy_feature_size,
                net_width=net_width)
    
    while True:

        envs = []
        for _ in range(threads):
            topo = np.random.choice(len(TOPOLOGIES))
            topo = TOPOLOGIES[topo]
            
            num_switches = np.random.choice(np.arange(MIN_GRAPH_SIZE, MAX_GRAPH_SIZE))
            
            num_flows = min((num_switches ** 2) / 2, MAX_FLOWS)
            num_flows = np.random.choice(np.arange(MIN_FLOWS, num_flows))

            logging.debug('TOPO: %s, NUM_FLOWS: %d, NUM_SWITCHES: %d' % topo, num_flows, num_switches)

            env = Environment(brain, num_flows, num_switches, topo, 
                            eps_start=eps_start, eps_end=eps_end,
                            eps_steps=eps_steps, thread_delay=thread_delay,
                            per_instance_limit=per_instance_limit, 
                            training_instance_limit=training_instance_limit, 
                            testing_instance_limit=testing_instance_limit,
                            test=test)
            envs.append(env)

        opts = [Optimizer(brain) for _ in range(optimizers)]

        for o in opts:
            o.start()

        for e in envs:
            e.start()

        for e in envs:
            e.join()

        for o in opts:
            o.stop()
        
        for o in opts:
            o.join()
        
        break # ??
        
        grad = 0.0
        count = 0
        for o in opts:
            grad += o.grad * o.count
            count += o.count

        grad = grad / count
        two_norm_grad = np.sqrt(np.sum([np.sum(q) for q in grad ** 2]))
        
        if two_norm_grad < GRAD_NORM_STOP:
            break

    training_instances = 0
    training_deviation = 0.0
    for e in envs:
        training_instances = training_instances + e.num_instances
        training_deviation = training_deviation + e.deviation
    
    avg_training_deviation = training_deviation/(training_instances*1.0)

    logging.info("TRAINING PHASE ENDED.")
    logging.info("AVG TRAIN DEVIATION: %f" % avg_training_deviation)

    return test_model(threads=threads, eps_start=eps_start, eps_end=eps_end,
                        eps_steps=eps_steps, thread_delay=thread_delay,
                        per_instance_limit=per_instance_limit, 
                        training_instance_limit=training_instance_limit, 
                        testing_instance_limit=testing_instance_limit)

def test_model(threads=THREADS, eps_start=EPS_START, eps_end=EPS_END,
                eps_steps=EPS_STEPS, thread_delay=THREAD_DELAY,
                per_instance_limit=PER_INSTANCE_LIMIT, 
                training_instance_limit=TRAINING_INSTANCE_LIMIT, 
                testing_instance_limit=TESTING_INSTANCE_LIMIT):
    test = True
    
    Environment.INSTANCE_NUM = itertools.count()

    envs = []
    for _ in range(threads):
        topo = np.random.choice(len(TOPOLOGIES))
        topo = TOPOLOGIES[topo]
        num_switches = np.random.choice(np.arange(MIN_GRAPH_SIZE, MAX_GRAPH_SIZE))
        num_flows = np.inf
        while 2 * num_flows > num_switches ** 2:
            num_flows = np.random.choice(MAX_FLOWS)

        env = Environment(brain, num_flows, num_switches, topo, 
                        eps_start=eps_start, eps_end=eps_end,
                        eps_steps=eps_steps, thread_delay=thread_delay,
                        per_instance_limit=per_instance_limit, 
                        training_instance_limit=training_instance_limit, 
                        testing_instance_limit=testing_instance_limit,
                        test=test)
        envs.append(env)
    test_instances = 0
    test_deviation = 0.0
    for e in envs:
        e.start()
    for e in envs:
        e.join()
    for e in envs:
        test_instances = test_instances + e.num_instances
        test_deviation = test_deviation + e.deviation

    avg_test_deviation = test_deviation/(test_instances*1.0)
    logging.info("AVG TEST DEVIATION: %f"  % avg_test_deviation)

    return avg_test_deviation

def delete():
    global brain

    if brain is not None:
        brain.session.close()
        tf.reset_default_graph()
        del brain

if __name__ == '__main__':
    # logger = logging.getLogger('root')
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(format=FORMAT)
    logging.basicConfig(level=logging.DEBUG)
    main()