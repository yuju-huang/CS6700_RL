from enum import IntEnum
import numpy as np
import random
from datetime import datetime

class Action(IntEnum):
    NONE = 0
    SCALE_UP = 1
    SCALE_DOWN = 2

    def random():
        random.seed(datetime.now())
        return Action(random.randint(int(Action.NONE), int(Action.SCALE_DOWN)))

class Reward:
    def __init__(self, r):
        self.reward = r

class State:
    num_lats = 3
    p50_idx = 0
    p95_idx = 1
    p99_idx = 2

    def __init__(self, cpu_util, lats):
        assert (isinstance(cpu_util, float))
        self.cpu_util = cpu_util
        assert (len(lats) == self.num_lats)
        self.lats = lats

    def __str__(self):
        return ("cpu_util=" + str(self.cpu_util) +
                ", lat[50, 95, 99]=[" + str(self.p50_lat()) + ", "
                                      + str(self.p95_lat()) + ", "
                                      + str(self.p99_lat()) + "]")

    def p50_lat(self):
        return self.lats[self.p50_idx]

    def p95_lat(self):
        return self.lats[self.p95_idx]

    def p99_lat(self):
        return self.lats[self.p99_idx]

    def np_vector(self):
        return np.array([[self.cpu_util, self.p50_lat(), self.p95_lat(), self.p99_lat()]])

def test_random():
    print(Action.random())

if __name__ == "__main__":
    test_random()
