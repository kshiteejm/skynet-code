#-- constants
ENV = 'Skynet-v0'

RUN_TIME = 60
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99
N_STEP_RETURN = 1
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.0
EPS_END  = 0.0
EPS_STEPS = 20000

MIN_BATCH = 32
LEARNING_RATE = 5e-2

LOSS_V = .5         # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

PER_INSTANCE_LIMIT = 100

MODEL_VERSION = 1   # state-dependent

TOPOLOGY = 'fat_tree'


TOPO_FEAT = True

DEBUG = True
VERBOSE = True
TESTING = True

TRAINING_INSTANCE_LIMIT = 10000 
TESTING_INSTANCE_LIMIT = 1000

_env = gym.make(ENV)
_env.__init__(topo_size=4, num_flows=1, topo_style='fat_tree', deterministic=True)
OBSERVATION_SPACE = _env.observation_space
ACTION_SPACE = _env.action_space
NULL_STATE = _env.get_null_state()

del _env