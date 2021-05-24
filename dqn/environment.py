import time

from env_xapian import Xapian
from rl import State
from rl import Action
from rl import Reward

class Environment:
    def __init__(self, actor, lat_weight, util_weight, p99_qos):
        self.actor = actor

        self.p99_qos = p99_qos 
        self.lat_weight = lat_weight
        self.util_weight = util_weight
    
        # Cache states from the previous action to prevent workload finishes
        # without reporting states.
        self.state_before_done = None
        self.reward_before_done = None

    def reset(self):
        time.sleep(2)
        self.start()
        self.state_before_done = None
        self.reward_before_done = None
        state = self.getStateVector()
        while state is None:
            state = self.getStateVector()
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
        # Do action
        self.actor.doAction(action)

        # Collect new state
        state = self.getState()
        if state is None:
            return self.state_before_done, self.reward_before_done, True, None

        # Calculate reward using performance QoS and resource utlization
        reward = self.reward(state)

        self.state_before_done = state.np_vector()
        self.reward_before_done = reward
        return state.np_vector(), reward, False, None

    # TODO: The reward function should be in evn_xapian
    def reward(self, state):
        return self.rewardImpl(state, self.actor.getMaxCPUShare())

    def rewardImpl(self, state, max_cpu):
        r = 0
        if (state.p99_lat() > self.p99_qos):
            r += (-1) * self.lat_weight * (state.p99_lat() / self.p99_qos) * 0.1
        else:
            r += 1 * self.lat_weight

        cpu_util = state.cpu_util / max_cpu
        r += (1 - cpu_util) * self.util_weight
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

if __name__ == "__main__":
    test()
