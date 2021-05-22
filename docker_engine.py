import docker
import os
import re
import subprocess
import time

client = docker.from_env()

class DockerEngine:
    @staticmethod
    def run(arg, name=None):
        dockerCmd = "docker run "
        if (name != None):
            dockerCmd += "--name " + name + " "
        dockerCmd += arg
        print("DockerEngine::run " + dockerCmd);
        pid = subprocess.Popen(dockerCmd, shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).pid
        os.system('stty sane')
        return pid

    def getCPUUtil(name):
#        container = client.containers.get(name)
#        status = container.stats(decode=True, stream=True)
#        print(status)
        cmd = "docker stats --format \"{{.CPUPerc}}\" --no-stream " + name
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        out = proc.communicate()
        os.system('stty sane')
        r = re.findall(r"\d+\.\d+%", str(out[0]))
        assert len(r) == 1
        return r[0]

    def getCPUUtil1(name):
        start = time.perf_counter()
        container = client.containers.get(name)
        status = container.stats(decode=False, stream=False)
        print("get CPU util takes " + str((time.perf_counter() - start) * 1000) + " ms")
        print(status)

    def setCPUShare(name, share):
#        container = client.containers.get(name)
#        container.update(cpu_shares=share)
        cmd = "docker update " + name + " --cpus " + str(share)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    
def test():
    DockerEngine.setCPUShare("frosty_golick", 5)

if __name__ == "__main__":
    test()
