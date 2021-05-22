import time
import os
import sysv_ipc as ipc
import subprocess
import struct

from datetime import datetime
from docker_engine import DockerEngine
from rl import State
from rl import Action

class Xapian:
    min_cpu_share = 1
    max_cpu_share = 8
    scale_factor = 2
    mq_path = "/tmp"
    mq_prj_id = 2333
    mq_cmd_get_lat = 8
    mq_cmd_put_lat = 18

    def __init__(self):
        # -- Container info
        self.server_name = "xapian-" + datetime.now().strftime("%H.%M.%S")
        self.cpu_share = 1
        # -- Message queue info
        self.mq_key = ipc.ftok(self.mq_path, self.mq_prj_id)
        self.mq = ipc.MessageQueue(self.mq_key, ipc.IPC_CREAT)

    def start(self):
        self.__startServer()
        self.__startClient()

    def getState(self):
        cpuUtil = DockerEngine.getCPUUtil(self.server_name)
        lats = self.__get_lats()
        return State(cpuUtil, lats)

    def doAction(self, action):
        if (action == Action.SCALE_UP) and (self.cpu_share < self.max_cpu_share):
            self.cpu_share *= self.scale_factor
        elif (action == Action.SCALE_DOWN) and (self.cpu_share > self.min_cpu_share):
            self.cpu_share /= self.scale_factor
        else:
            return
        DockerEngine.setCPUShare(self.server_name, self.cpu_share)

    def getMaxCPUShare(self):
        return self.max_cpu_share

    # ----- Internal APIs for test only
    def setServerName(self, name):
        self.server_name = name

    # ----- Internal implementation -----
    def __startServer(self):
        arg = "-it -p 3366:3366 --entrypoint /TailBench/xapian/run_server.sh tailbench 8"
        DockerEngine.run(arg, self.server_name)
        time.sleep(3)
        DockerEngine.setCPUShare(self.server_name, self.cpu_share)
        print("done start server")

    def __startClient(self):
        cmd = "/home/yh885/TailBench/xapian/run_client.sh 10000 8 1"
        self.__shell_run(cmd)
        print("done start client")

    def __shell_run(self, cmd):
        pid = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).pid
        os.system('stty sane')
        return pid

    def __get_lats(self):
        msg_string = "GET LAT\0"
        self.mq.send(msg_string, block=True, type=self.mq_cmd_get_lat)

        recv_buf, recv_type = self.mq.receive(block=True, type=self.mq_cmd_put_lat)
        assert (recv_type == self.mq_cmd_put_lat)
        # In the format of double[3] for p50, p95, p99 latencies.
        return struct.unpack("ddd", recv_buf)

def test_doAction():
    x = Xapian()
    name = "frosty_golick"
    x.setServerName(name)
    x.doAction(Action.SCALE_UP)
    time.sleep(1)
    x.doAction(Action.SCALE_UP)
    time.sleep(1)
    x.doAction(Action.SCALE_UP)
    time.sleep(1)
    x.doAction(Action.SCALE_UP)
    time.sleep(1)
    x.doAction(Action.SCALE_UP)
    print("done scale_up, sleep 3 sec")
    time.sleep(3)

    x.doAction(Action.SCALE_DOWN)
    time.sleep(1)
    x.doAction(Action.SCALE_DOWN)
    time.sleep(1)
    x.doAction(Action.SCALE_DOWN)
    time.sleep(1)
    x.doAction(Action.SCALE_DOWN)
    time.sleep(1)
    x.doAction(Action.SCALE_DOWN)
    print("done scale_down, sleep 3 sec")
    time.sleep(3)
    
if __name__ == "__main__":
    test_doAction()
