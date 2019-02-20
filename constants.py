import logging
import os
import gym
import gym_skynet
from deepwalk import get_deepwalk_representation
import numpy as np

#-- constants
ENV = 'Skynet-v0'

RUN_TIME = 60
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99
N_STEP_RETURN = 1
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.5
EPS_END  = 0.0
EPS_STEPS = 50000

MIN_BATCH = 32
LEARNING_RATE = 5e-2

LOSS_V = .5         # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

PER_INSTANCE_LIMIT = 10

MODEL_VERSION = 1   # state-dependent

TOPOLOGIES = ['fat_tree']
MIN_GRAPH_SIZE = 2
MAX_GRAPH_SIZE = 4
MIN_FLOWS = 2
MAX_FLOWS = 4

# TOPO_FEAT = True

TESTING = False

TRAINING_INSTANCE_LIMIT = 10000
TESTING_INSTANCE_LIMIT = 1000

GRAD_NORM_STOP = 0.1

NODE_FEATURE_SIZE = 16
GNN_ROUNDS = 4
NET_WIDTH = 16
POLICY_FEATURE_SIZE = 16

ISOLATION_PROJ = None
FLOW_ID_PROJ = None
REACHABILITY_PROJ = None

if os.path.isfile('isolation_proj.npy'):
    ISOLATION_PROJ = np.load('isolation_proj.npy')
else:
    ISOLATION_PROJ = np.random.normal(0.0, 1.0, (POLICY_FEATURE_SIZE, MAX_FLOWS))
    np.save('isolation_proj.npy', ISOLATION_PROJ)

if os.path.isfile('flow_id_proj.npy'):
    FLOW_ID_PROJ = np.load('flow_id_proj.npy')
else:
    FLOW_ID_PROJ = np.random.normal(0.0, 1.0, (POLICY_FEATURE_SIZE, MAX_FLOWS))
    np.save('flow_id_proj.npy', FLOW_ID_PROJ)

if os.path.isfile('reachability_proj.npy'):
    REACHABILITY_PROJ = np.load('reachability_proj.npy')
else:
    REACHABILITY_PROJ = np.random.normal(0.0, 1.0, (POLICY_FEATURE_SIZE, MAX_FLOWS))
    np.save('reachability_proj.npy', REACHABILITY_PROJ)
if os.path.isfile('isolation_proj.npy'):
    ISOLATION_PROJ = np.load('isolation_proj.npy')
else:
    POLICY_PROJ = np.random.normal(0.0, 1.0, ())


class Colorize:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHT_PURPLE = '\033[94m'
    PURPLE = '\033[95m'
    END = '\033[0m'

    @classmethod
    def highlight(cls, s):
        return ("\x1b[6;30;42m" + s + "\x1b[0m")

    @classmethod
    def red(cls, s):
        return (cls.RED + s + cls.END)

    @classmethod
    def green(cls, s):
        return (cls.GREEN + s + cls.END)

    @classmethod
    def yellow(cls, s):
        return (cls.YELLOW + s + cls.END)

    @classmethod
    def lightPurple(cls, s):
        return (cls.LIGHT_PURPLE + s + cls.END)

    @classmethod
    def purple(cls, s):
        return (cls.PURPLE + s + cls.END)
