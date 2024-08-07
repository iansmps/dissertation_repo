
import matlab.engine
import numpy as np
import h5py
from random import seed
from random import random

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

#from xvfbwrapper import Xvfb

#This environment calculates reward based on the ratio of samples below the upper bound
class CassandraEnv(py_environment.PyEnvironment):

    def __init__(self, config):
        #The action is the desired number of active nodes in the cluster
        #self._action_spec = array_spec.BoundedArraySpec(
        #shape=(), dtype=np.int32, minimum=0, maximum=19, name='action')

	    #The action represents a delta in the current number of active nodes
	    # 0->decrease nodes by 1
	    # 1->keep nodes
	    # 2->increase nodes by 1
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        #The observations consists of [throughput, active nodes, a node is joining or leaving, delay mean, delay median, error delay q95]
        self._observation_spec = {'observations': array_spec.BoundedArraySpec(shape=(7,), dtype=np.float32, minimum=[0, 3, -1, 0, 0, -100,-100], maximum=[100000, 9, 1, 100, 100, 100, 100], name='observation'),
                                  'legal_moves': array_spec.ArraySpec(shape=(3,), dtype=np.bool_, name='legal_moves')}

        #self._vdisplay = Xvfb()
        #self._vdisplay.start()

        #Connect to matlab and load the simulation
        #self._eng=matlab.engine.connect_matlab()
        self._eng = matlab.engine.start_matlab()
        self._eng.load_system(config.get('SimulinkModel'))

        self._config = config

        #set the configuration
        
        self._eng.SetSimulationConfig(self._config.get('SimulinkModel'), self._config.getint('BootDelay'), self._config.getint('InitialNodes'), self._config.getint('AverageTimeWindow'), self._config.getfloat('ResponseTimeUpperBound'), nargout=0)
        self._step_delay = config.getfloat('StepDelay')
        self._sync_delay = config.getfloat('SyncDelay')
        self._next_step = self._step_delay
        self._next_sync = self._sync_delay*2
        self._target = config.getfloat('ResponseTimeUpperBound')
        self._maxthr = config.getint('MaxThr')
        self._prev_lat = 0
        #Set that the simulation is not started
        self._sim_on = False

        seed()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec    

    def get_metrics_logfile(self):
        #Returns execution metrics from the logfile
        #Returns an array with: [delay mean, delay median, delay q95, delay q99, delay max]

        metrics = [0, 0, 0, 0, 0, 0]

        f = h5py.File('responseTime.mat','r')
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        data = np.array(data)

        #To solve a bug which happens some time where the file is not ready
        print(data.shape)
        
        if (len(data) > 1 and data.shape[1] == 2):
            differ = self._prev_lat - np.quantile(data[-200:,1], 0.95)
            metrics = [np.mean(data[-200:,1]), np.median(data[-200:,1]), np.quantile(data[-200:,1], 0.95), np.quantile(data[-200:,1], 0.99), np.max(data[-200:,1]), differ]
            self._prev_lat = metrics[2]

        return metrics

    def get_observation(self):
        #Obtain metrics from the simulator and from the logfile
        #Returns an array with: [throughput, active nodes, delay mean, delay median, delay q95, delay q99, delay max]

        # para obtener throughput y active nodes en el segundo x
        sim_metrics = self._eng.GetDataSync(self._config.get('SimulinkModel'), self._next_step, self._next_sync, nargout=2)

        log_metrics = self.get_metrics_logfile()

        return np.concatenate((sim_metrics, log_metrics))

    def set_nodes_get_observation(self, next_nodes):
        #Set the number of active nodes in the simulatior and obtain metrics from the simulator and from the logfile
        #Returns an array with: [throughput, active nodes, delay mean, delay median, delay q95, delay q99, delay max]

        sim_metrics = self._eng.SetActiveNodesAndGetDataSync(self._config.get('SimulinkModel'), self._next_step, self._next_sync, int(next_nodes), nargout=2)

        log_metrics = self.get_metrics_logfile()

        return np.concatenate((sim_metrics, log_metrics))


    def create_observation(self, metrics):
        #Creates the observation to pass to the model
        #The received metrics are: [throughput, active nodes, delay mean, delay median, delay q95, delay q99, delay max]

        #The observation returned is an array with: [throughput, active nodes, a node is joining or leaving, delay mean, delay median, error delay q95]
        
        #Save the current number of nodes
        self._active_nodes = metrics[1]

        error = metrics[4] - self._target

        n1 = metrics[0]/self._maxthr 
        n2 = round(metrics[1]/10,1)
        n4 = metrics[2] if (metrics[2]<=1).all() else 1
        n5 = metrics[3] if metrics[3] <= 1 else 1
        n6 = error if error <= 1 else 1
        n7 = -1 if metrics[7] < -1 else 1 if metrics[7] > 1 else metrics[7]



        #Create the observation
        observation = np.array([n1, n2, self._expected_nodes - self._active_nodes, n4, n5, n6, n7], dtype=np.float32)
        print(n2)

        if observation[2] == 0: #if no nodes starting or stopping just check if not in the max or min
            if (self._active_nodes > self._config.getint('MinNodes')) and (self._active_nodes < self._config.getint('MaxNodes')):
                legal_moves = np.array([True, True, True], dtype=np.bool_)
            elif self._active_nodes == self._config.getint('MinNodes'):
                legal_moves = np.array([False, True, True], dtype=np.bool_)
            else:
                legal_moves = np.array([True, True, False], dtype=np.bool_)
        else: 
            legal_moves = np.array([False, True, False], dtype=np.bool_)

        observations_and_legal_moves = {'observations': observation,
                                        'legal_moves': legal_moves}

        return observations_and_legal_moves

    def _reset(self):
        #Restart simulation and return TimeStep with initial state
        #Call to restart simulation
        if self._sim_on:
            self._eng.ResetSimulation(self._config.get('SimulinkModel'), self._config.getint('SyncDelay'), self._config.getint('InitialNodes'), self._config.getfloat('ResponseTimeUpperBound'), nargout=0)
        else:
            self._eng.StartSimulation(self._config.get('SimulinkModel'), self._config.getint('SyncDelay'), self._config.getint('InitialNodes'), self._config.getfloat('ResponseTimeUpperBound'), nargout=0)
            self._sim_on = True

        #Restart the time counter
        self._next_step = self._step_delay
        self._next_sync = self._sync_delay*2

        #Restart the active and expected nodes
        self._active_nodes = self._config.getint('InitialNodes')
        self._expected_nodes = self._active_nodes

        #Get observation
        values = self.get_observation()

        observations_and_legal_moves = self.create_observation(values)

        #Increment time counter
        self._next_step += self._step_delay
        self._next_sync += self._sync_delay
    
        return ts.restart(observations_and_legal_moves)

    def _step(self, action):

        ###There is no last state in this environment###

        if not self._sim_on:
            self._reset()

	    #Check if simulation is running
        status = self._eng.get_param(self._config.get('SimulinkModel'), 'SimulationStatus')
        if status == 'stopped':
            self._eng.eval('exception = MException.last;', nargout=0)
            print(self._eng.eval('getReport(exception)'))
            print("***Simulation stopped unexpectedly***")
            self._reset()

        #Apply action to the simulation and obtain state
        #Only change node if action not 1
        if action == 1:
            values = self.get_observation()
        else:
            next_nodes = self._active_nodes + int(action) - 1
            values = self.set_nodes_get_observation(next_nodes)
            self._expected_nodes = next_nodes

        observations_and_legal_moves = self.create_observation(values)

        #calculate reward
        error = observations_and_legal_moves['observations'][5]
        #reward = (-((self._active_nodes - 3) / 6) * 0.5) if error <= 0 else -1
        reward = (10-((self._active_nodes - 3) * 0.75)) if error <= 0 else 0
        ##reward = (1-((self._active_nodes - 3) * 0.075)) if error <= 0 else 0
        #reward = (1 - self._active_nodes * 0.1) if error <= 0 else -1
        #Increment time counter
        self._next_step += self._step_delay
        self._next_sync += self._sync_delay
        
        #End simulation if response time diverges
        #From experimentation we have found that when response time increseas too much it is difficult to recover
        if (observations_and_legal_moves['observations'][3] > self._config.getint('TerminationMaxThreshold')):
            return ts.termination(observations_and_legal_moves, reward=reward)
        else:
            return ts.transition(observations_and_legal_moves, reward=reward, discount=1.0)

    def __del__(self):
        self._eng.quit()
        #self._vdisplay.stop()
