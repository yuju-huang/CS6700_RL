import time

from env_xapian import Xapian
from rl import State
from rl import Action
from rl import Reward

class Environment:
    # The weight for calculating reward.
    p99_qos = 10 # 10 ms as qos violation
    lat_weight = 0.8
    util_weight = 0.2
    
    def __init__(self, actor):
        self.actor = actor

    def start(self):
        # Run server and start workloads
        self.actor.start()

    def finish(self):
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
        # Do action
        self.actor.doAction(action)

        # Collect new state
        state = self.getState()
        if state is None:
            return None, None

        # Calculate reward using performance QoS and resource utlization
        return self.reward(state), state.np_vector()

    # TODO: The reward function should be in evn_xapian
    def reward(self, state):
        return Environment.rewardImpl(state, self.actor.getMaxCPUShare())

    def rewardImpl(state, max_cpu):
        r = 0
        if (state.p99_lat() > Environment.p99_qos):
            r += (-1) * Environment.lat_weight * (state.p99_lat() / Environment.p99_qos)
        else:
            r += 1 * Environment.lat_weight

        cpu_util = state.cpu_util / max_cpu
        r += (1 - cpu_util) * Environment.util_weight
        return r

def dump(reward, state):
    print("state=" + str(state))
    print("reward=" + str(reward))

def test():
    e = Environment(Xapian())
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

def test_reward():
    s1 = State("500%", [5, 5, 5])
    s2 = State("200%", [5, 5, 15])
    
    r1 = Environment.rewardImpl(s1, 8)
    r2 = Environment.rewardImpl(s2, 8)

    print("r1=" + str(r1))
    print("r2=" + str(r2))

    assert (r1 == (1 * Environment.lat_weight) + Environment.util_weight * float(3/8))
    assert (r2 == (-1 * (s2.p99_lat() / Environment.p99_qos) * Environment.lat_weight) + Environment.util_weight * float(6/8))

if __name__ == "__main__":
    test_reward()
    test()
