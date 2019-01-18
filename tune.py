from __future__ import division, print_function
import sys
import numpy as np
from hyperopt import fmin, tpe, hp, pyll

import skynet_A3C 

SPACE = {
    'GAMMA': hp.uniform('GAMMA', 0.9, 1.0),
    'N_STEP_RETURN': hp.choice('N_STEP_RETURN', list(range(1,13))),
    'LEARNING_RATE': hp.loguniform('LEARNING_RATE', np.log(1e-3), np.log(0.1)),
    'MIN_BATCH': hp.choice('MIN_BATCH', [32, 64, 128, 256]),
    # 'LOSS_V': hp.uniform('LOSS_V', ),
    # 'LOSS_ENTROPY': hp.uniform('LOSS_ENTROPY', ),
    'EPS_START': hp.uniform('EPS_START', 0.3, 0.7),
    'EPS_STOP': hp.uniform('EPS_STOP', 0.0, 0.3),
    'EPS_STEPS': hp.uniform('EPS_STEPS', [i * 10000 for i in range(6)])
}


def run(space):
    val = skynet_A3C.main(
                gamma=space['GAMMA'],
                n_step_return=space['N_STEP_RETURN'],
                learning_rate=space['LEARNING_RATE'],
                min_batch=space['MIN_BATCH'],
                # loss_v=space['LOSS_V'],
                # loss_entropy=space['LOSS_ENTROPY'],
                eps_start=space['EPS_START'],
                eps_end=space['EPS_STOP'],
                eps_steps=space['EPS_STEPS']
            )
    return val



if __name__ == '__main__':
    best = fmin(fn=run,
                space=SPACE,
                algo=tpe.suggest,
                max_evals=100)
    print(best)