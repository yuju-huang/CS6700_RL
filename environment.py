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

    def getState(self):
        state = self.actor.getState()
        print(state.cpu_util)
        print(state.lats)

    def step(self, action):
        # Do action

        # Collect new state

        # Calculate reward using performance QoS and resource utlization

        return Reward(0), State(0, 500)

    def reward(self, state):
        return rewardImpl(state, self.actor.getMaxCPUShare())

    def rewardImpl(state, max_cpu):
        r = 0
        if (state.p99_lat() > Environment.p99_qos):
            r += (-1) * Environment.lat_weight
        else:
            r += 1 * Environment.lat_weight

        cpu_util = state.cpu_util / max_cpu
        r += (1 - cpu_util) * Environment.util_weight
        return r

def test():
    e = Environment(Xapian())
    e.start()
    while True:
        time.sleep(0.01)
        e.getState()

def test_reward():
    s1 = State("500%", [5, 5, 5])
    s2 = State("200%", [5, 5, 15])
    
    r1 = Environment.rewardImpl(s1, 8)
    r2 = Environment.rewardImpl(s2, 8)

    assert (r1 == (1 * Environment.lat_weight) + Environment.util_weight * float(3/8))
    assert (r2 == (-1 * Environment.lat_weight) + Environment.util_weight * float(6/8))

if __name__ == "__main__":
    #test()
    test_reward()
