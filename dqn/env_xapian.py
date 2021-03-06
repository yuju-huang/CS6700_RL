import time
import os
import psutil
import sysv_ipc as ipc
import subprocess
import struct

from datetime import datetime
from docker_engine import DockerEngine
from rl import State
from rl import Action

class Xapian:
    min_cpu_share = 1
    #max_cpu_share = 8
    max_cpu_share = 16
    max_latency = 100
    min_latency = 1
    num_lats = 3
    scale_factor = 2
    mq_path = "/tmp"
    mq_prj_id = 2333
    mq_cmd_get_lat = 3
    mq_cmd_put_lat = 2
    mq_cmd_finish = 1
    get_util_interval = 0.3

    def __init__(self, workload_path, lat_weight, util_weight, p99_qos):
        # -- Container info
        self.server_name = ""
        self.cpu_share = 1
        # -- Message queue info
        self.mq_key = None
        self.mq = None
        self.num_cpus = os.cpu_count()
        self.workload_file = workload_path

        # Cache states from the previous action to prevent workload finishes
        # without reporting states.
        self.state_before_done = None
        self.reward_before_done = None

        self.p99_qos = p99_qos 
        self.lat_weight = lat_weight
        self.util_weight = util_weight
        # The unit in reward range [-1, 1]
        self.lat_unit = (float)(2) / ((Xapian.max_latency - Xapian.min_latency + 1) / self.p99_qos)
        # The cpu_share has only 4 levels: 1, 2, 4, 8, 16 so there are 4 intervals
        self.util_unit = (float)(2) / 4 #(Xapian.max_cpu_share - Xapian.min_cpu_share + 1)
        print("self.lat_unit=", self.lat_unit, ", self.util_unit=", self.util_unit)

    def start(self):
        print("Xapian start")
        self.state_before_done = None
        self.reward_before_done = None
        self.startServer()
        self.startClient()

    def isRunning(self):
        cmd = "ps aux | grep xapian_networked_client | grep -v grep | awk '{print $2}'"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        out = proc.communicate(timeout=1)
        os.system('stty sane')
        if (out == None):
            return False
        return True
        
    def finish(self):
        cmd = "ps aux | grep xapian_networked_client | grep -v grep | awk '{print $2}' | xargs -I% sudo kill %"
        self.__shell_run(cmd)
        cmd = "sudo ipcrm --all=msg"
        self.__shell_run(cmd)
        print("Xapian done")

    def getState(self):
        # The CPU utilization from docker is so slow, it takes more than 1 sec.
        #c = DockerEngine.getCPUUtil(self.server_name)
        # cpu_util's format like 635.76% 
        #cpuUtil = float(c.strip('%')) / 100

        # Format like 6.2, which is a util for all cpus so need to scale it by num_cpus
        #cpuUtil = float(psutil.cpu_percent(Xapian.get_util_interval) * self.num_cpus) / 100
        cpuUtil = (float)(self.cpu_share)
        lats = self.__get_lats()
        if (lats == None):
            return None

        assert (len(lats) == self.num_lats)
        return State(cpuUtil, lats)

    def step(self, action):
        # Do action
        succ = self.doAction(action)
        # This tries to clear the latency stats
        garbage = self.__get_lats()
        if garbage is None:
            return self.state_before_done, self.reward_before_done, True, None

        time.sleep(0.2)
        # Collect new state
        state = self.getState()
        if state is None:
            return self.state_before_done, self.reward_before_done, True, None

        # Calculate reward using performance QoS and resource utlization
        reward = self.reward(state, succ)

        self.state_before_done = state
        self.reward_before_done = reward
        return state, reward, False, None

    def doAction(self, action):
        if (action == Action.NONE):
             return True

        if (action == Action.SCALE_UP) and (self.cpu_share < self.max_cpu_share):
            self.cpu_share *= self.scale_factor
        elif (action == Action.SCALE_DOWN) and (self.cpu_share > self.min_cpu_share):
            self.cpu_share /= self.scale_factor
        else:
            # Return False to indicate the action fails. This pushes the model to do Action.NONE.
            return False
        DockerEngine.setCPUShare(self.server_name, self.cpu_share)
        return True

    def reward(self, state, action_succ):
        assert state is not None

        if action_succ == False:
            return -0.2

        lat_reward = 0
        if (state.p99_lat() > self.p99_qos):
            lat_reward -= ((state.p99_lat() - self.p99_qos) / self.p99_qos) * 0.1
            #lat_reward -= ((state.p99_lat() - self.p99_qos) / self.p99_qos) * self.lat_unit
        else:
            lat_reward = 1

        #util_reward = 1
        #util_reward -= (state.cpu_util - Xapian.min_cpu_share) * self.util_unit
        util_reward = self.util_reward(state.cpu_util)

        return lat_reward * self.lat_weight + util_reward * self.util_weight

    def util_reward(self, cpu_util):
        if cpu_util == self.min_cpu_share:
            return 1
        elif cpu_util == 2 * self.min_cpu_share:
            return 0.5
        elif cpu_util == 4 * self.min_cpu_share:
            return 0
        elif cpu_util == 8 * self.min_cpu_share:
            return -0.5
        else:
            assert cpu_util == self.max_cpu_share
            return -1

    def getMaxCPUShare(self):
        return self.max_cpu_share

    def getMaxLatency(self):
        return self.max_latency

    # ----- Internal APIs for test only
    def setServerName(self, name):
        self.server_name = name

    # ----- Internal implementation -----
    def startServer(self):
        self.server_name = "xapian-" + datetime.now().strftime("%H.%M.%S")
        arg = "-it -p 3366:3366 --entrypoint /TailBench/xapian/run_server.sh yh885/tailbench 8"
        DockerEngine.run(arg, self.server_name)
        time.sleep(3)
        DockerEngine.setCPUShare(self.server_name, self.cpu_share)
        print("done start server")

    def startClient(self):
        cmd = "/home/yh885/TailBench/xapian/run_client.sh 10000 8 1 " + self.workload_file + " > client.log 2>&1"
        self.__shell_run(cmd)
        self.mq_key = ipc.ftok(self.mq_path, self.mq_prj_id, silence_warning=True)
        self.mq = ipc.MessageQueue(self.mq_key, ipc.IPC_CREAT)
        print("done start client")

    def __shell_run(self, cmd):
        pid = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).pid
        os.system('stty sane')
        return pid

    def __get_lats(self):
        msg_string = "GET LAT\0"
        self.mq.send(msg_string, block=True, type=self.mq_cmd_get_lat)

        #recv_buf, recv_type = self.mq.receive(block=True, type=self.mq_cmd_put_lat)
        recv_buf, recv_type = self.mq.receive(block=True, type=(-1*self.mq_cmd_put_lat))
        if (recv_type == Xapian.mq_cmd_finish):
            return None

        assert (recv_type == self.mq_cmd_put_lat)
        # In the format of double[3] for p50, p95, p99 latencies.
        return struct.unpack("ddd", recv_buf)

def test_doAction():
    x = Xapian()
    name = "yo"
    x.setServerName(name)
    DockerEngine.setCPUShare(name, 1)
    x.startClient()
    time.sleep(5)
    print("Original state=", x.getState())
    x.doAction(Action.SCALE_UP)
    print("SCALE_UP, state=", x.getState())
    x.doAction(Action.SCALE_UP)
    print("SCALE_UP, state=", x.getState())
    x.doAction(Action.SCALE_UP)
    print("SCALE_UP, state=", x.getState())
    x.doAction(Action.SCALE_UP)
    print("SCALE_UP, state=", x.getState())
    x.doAction(Action.SCALE_UP)
    print("SCALE_UP, state=", x.getState())
    print("done scale_up, sleep 3 sec")
    time.sleep(3)

    x.doAction(Action.SCALE_DOWN)
    print("SCALE_DOWN, state=", x.getState())
    x.doAction(Action.SCALE_DOWN)
    print("SCALE_DOWN, state=", x.getState())
    x.doAction(Action.SCALE_DOWN)
    print("SCALE_DOWN, state=", x.getState())
    x.doAction(Action.SCALE_DOWN)
    print("SCALE_DOWN, state=", x.getState())
    x.doAction(Action.SCALE_DOWN)
    print("done scale_down, sleep 3 sec")
    time.sleep(3)

def test_getState():
    while True:
        print(psutil.cpu_percent(0.3))

if __name__ == "__main__":
    #test_getState()
    test_doAction()
