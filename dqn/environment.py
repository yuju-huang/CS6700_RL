import time

from env_xapian import Xapian
from rl import State
from rl import Action
from rl import Reward

class Environment:
    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        time.sleep(2)
        self.start()
        state = self.getState()
        while state is None:
            state = self.getState()
        return state 

    def start(self):
        # Run server and start workloads
        self.actor.start()

    def close(self):
        self.actor.finish()

    def isRunning(self):
        return self.actor.isRunning()

    def getState(self):
        return self.actor.getState()

    def getStateVector(self):
        state = self.getState()
        if (state == None):
            return None
        return state.np_vector()

    def step(self, action):
        return self.actor.step(action)

def dump(reward, state):
    print("state=" + str(state))
    print("reward=" + str(reward))

def test():
    e = Environment(
        Xapian("/home/yh885/TailBench/xapian/workload_fix4s_20s.dec",
               lat_weight, util_weight, p99_qos))
    e.start()
    while True:
        r, s = e.step(Action.SCALE_UP)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_UP)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_UP)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_UP)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_DOWN)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_DOWN)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_DOWN)
        dump(r, s)
        time.sleep(1)
        r, s = e.step(Action.SCALE_DOWN)
        dump(r, s)
        time.sleep(1)

if __name__ == "__main__":
    test()
