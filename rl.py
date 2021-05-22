from enum import Enum

class Action(Enum):
    SCALE_UP = 1
    SCALE_DOWN = 2

class Reward:
    def __init__(self, r):
        self.reward = r

class State:
    num_lats = 3
    p50_idx = 0
    p95_idx = 1
    p99_idx = 2

    def __init__(self, cpu_util, lats):
        # cpu_util's format like 635.76% 
        self.cpu_util = float(cpu_util.strip('%')) / 100
        assert (len(lats) == self.num_lats)
        self.lats = lats

    def p50_lat(self):
        return self.lats[self.p50_idx]

    def p95_lat(self):
        return self.lats[self.p95_idx]

    def p99_lat(self):
        return self.lats[self.p99_idx]
