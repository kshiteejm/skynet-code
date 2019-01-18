from __future__ import division, print_function
import sys
import numpy as np
from hyperopt import fmin, tpe, hp, pyll

from skynet_A3C import Environment, Optimizer, Brain

SPACE = {
    'EPS_START': hp.uniform('EPS_START', 0.0, 1.0),
    'EPS_END': hp.uniform('EPS_END', 0.0, 1.0),
    'EPS_STEPS': hp.uniform('EPS_STEPS', 5, 20),
    'MIN_BATCH': hp.ch

}


def run(space):



if __name__ == '__main__':
    best = fmin(fn=run,
                space=SPACE,
                algo=tpe.suggest,
                max_evals=100)
    print(best)