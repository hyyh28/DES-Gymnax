import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from gymnax.environments.environment import TEnvParams
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
import timeit
from datetime import datetime, timedelta

"""
Machines are created here
Whenever you use the machine release it
"""
class Machine(object):
    def __init__(self, name):
        self.machineID = int(name)
        self.machineName = "Machine_"+str(name)
        self.machineBusy = False
        self.jobOverTime = 0
        self.now = 0
        self.jobsDone = []
    
    def processJob(self, jobID, time, pTime):
        # check if machine is busy or not
        assert self.machineBusy == False
        self.onJob = jobID
        self.now = time
        self.processTime = pTime
        # import pdb; pdb.set_trace()
        self.jobOverTime = self.now + self.processTime
        self.machineBusy = True
        return
    
    def releaseMachine(self):
        # check if currently in use
        assert self.machineBusy == True
        self.jobsDone.append(self.onJob)
        self.machineBusy = False
        return
    

class Job(object):
    def __init__(self, name):
        self.jobID = int(name)
        self.jobName = "Job_"+str(name)
        self.jobBusy = False
        self.processDetails = []
        self.noOfProcess = 0
        self.now = 0
        self.machineVisited = 0
    
    def getProcessed(self):
        assert self.jobBusy == False
        self.jobBusy = True
        return
    
    def releaseJob(self):
        assert self.jobBusy == True
        self.jobBusy = False
        return
    
@struct.dataclass
class EnvState(environment.EnvState):
    machine_status: jnp.ndarray
    job_visits: jnp.ndarray
    machine_busy_until: jnp.ndarray
    time: int
    jobs_processed: jnp.ndarray

@struct.dataclass
class EnvParams(environment.EnvParams):
    num_machines: int = 10
    num_jobs: int = 10
    machine_processing_time: jnp.ndarray = jnp.ones((10, 10)) * 10

class JobMarket(Environment):
    def __init__(self, num_machines: int, num_jobs: int, machine_processing_time: jnp.ndarray):
        self.num_machines = num_machines
        self.num_jobs = num_jobs
        self.machine_processing_time = machine_processing_time
        self.obs_shape = (num_machines + num_jobs + num_machines,)
        self.machines = [Machine(i) for i in range(num_machines)]
        self.jobs = [Job(i) for i in range(num_jobs)]
        super().__init__()

    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def getProcessTime(self, macID, jID):
        return self.machine_processing_time[macID, jID]
    
    def getEmptyMachines(self):
        return [i for i in range(self.num_machines) if not self.machines[i].machineBusy]
    
    def getBusyMachines(self):
        return [i for i in range(self.num_machines) if self.machines[i].machineBusy]
    
    def getEmptyJobs(self):
        return [i for i in range(self.num_jobs) if not self.jobs[i].jobBusy]
    
    def getBusyJobs(self):
        return [i for i in range(self.num_jobs) if self.jobs[i].jobBusy]
    