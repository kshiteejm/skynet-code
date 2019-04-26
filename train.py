import logging
import numpy as np
import random
import multiprocessing as mp
from agent_env import Agent, CentralAgent

from constants import LOG_FILE, RANDOM_SEED, NUM_AGENTS

def agent(agent_id, nnet_param_queue, train_queue):
    agent = Agent(agent_id, nnet_param_queue, train_queue)
    agent.run()

def central_agent(nnet_param_queues, train_queues):
    central_agent = CentralAgent(nnet_param_queues, train_queues)
    central_agent.run()

def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    nnet_param_queues = []
    train_queues = []
    for _ in range(NUM_AGENTS):
        nnet_param_queues.append(mp.Queue(1))
        train_queues.append(mp.Queue(1))
    
    coordinator = mp.Process(
        target=central_agent,
        args=(nnet_param_queues, train_queues)
    )
    coordinator.start()

    agents = []
    for agent_id in range(NUM_AGENTS):
        agents.append(
            mp.Process(
                target=agent,
                args=(
                    agent_id, 
                    nnet_param_queues[agent_id],
                    train_queues[agent_id])
            )
        )
        # We never start the agents?
    
    # wait till all epochs are over
    coordinator.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE)
    logging.getLogger().addHandler(fh)
    main()