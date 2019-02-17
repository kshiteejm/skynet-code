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

from constants import ADJ_MAT, EPS_END, EPS_START, EPS_STEPS, \
                    GAMMA, LEARNING_RATE, LOSS_ENTROPY, LOSS_V, MIN_BATCH, \
                    N_STEP_RETURN, OPTIMIZERS, PER_INSTANCE_LIMIT, TESTING, \
                    TESTING_INSTANCE_LIMIT, THREAD_DELAY, THREADS, TOPO_FEAT, \
                    TRAINING_INSTANCE_LIMIT, GRAD_NORM_STOP, NODE_FEATURES

brain = None

def main(gamma=GAMMA, n_step_return=N_STEP_RETURN, learning_rate=LEARNING_RATE, 
            min_batch=MIN_BATCH, loss_v=LOSS_V, loss_entropy=LOSS_ENTROPY,
            eps_start=EPS_START, eps_end=EPS_END, eps_steps=EPS_STEPS,
            topo_feat=TOPO_FEAT, thread_delay=THREAD_DELAY,
            threads=THREADS, optimizers=OPTIMIZERS,
            per_instance_limit=PER_INSTANCE_LIMIT,
            training_instance_limit=TRAINING_INSTANCE_LIMIT,                
            testing_instance_limit=TESTING_INSTANCE_LIMIT,
            test=TESTING):

    global brain
    # TODO fix    

    delete()

    node_features = NODE_FEATURES

    Environment.INSTANCE_NUM = itertools.count()

    brain = Brain(node_features, gamma=gamma, n_step_return=n_step_return, 
                learning_rate=learning_rate, min_batch=min_batch, loss_v=loss_v, 
                loss_entropy=loss_entropy, topo=topo_feat)
    
    while True:
        envs = [Environment(node_features, brain, render=False, 
                eps_start=eps_start, eps_end=eps_end, 
                eps_steps=eps_steps, thread_delay=thread_delay,
                per_instance_limit=per_instance_limit, 
                training_instance_limit=training_instance_limit, 
                testing_instance_limit=testing_instance_limit,
                test=test) for _ in range(threads)]
    
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
        break
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

    return test_model(node_features)

def test_model(node_features, 
        eps_start=EPS_START, eps_end=EPS_END, eps_steps=EPS_STEPS,
        thread_delay=THREAD_DELAY, threads=THREADS,
        per_instance_limit=PER_INSTANCE_LIMIT, 
        training_instance_limit=TRAINING_INSTANCE_LIMIT, 
        testing_instance_limit=TESTING_INSTANCE_LIMIT,
        test=TESTING):
    test = True
    Environment.INSTANCE_NUM = itertools.count()

    envs = [Environment(brain, render=False, node_features=None,
                eps_start=eps_start, eps_end=eps_end, 
                eps_steps=eps_steps, thread_delay=thread_delay,
                per_instance_limit=per_instance_limit, 
                training_instance_limit=training_instance_limit, 
                testing_instance_limit=testing_instance_limit,
                test=test) for _ in range(threads)]
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
    logging.basicConfig(level=logging.INFO)
    main()