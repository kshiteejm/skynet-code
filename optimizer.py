from __future__ import print_function

import logging
import threading

import numpy as np

class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain
        self.grad = None
        self.count = 0

    def run(self):
        while not self.stop_signal:
            grad, count = self.brain.optimize()
            if self.grad is None:
                self.grad = np.zeros_like(grad)
            self.grad += grad
            self.count += count
            
    def stop(self):
        self.stop_signal = True