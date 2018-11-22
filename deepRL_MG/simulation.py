#!/usr/bin/env python3
from __future__ import division
import numpy as np
import math

# choose the data you want to load
# data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')
data_complete = ['simulation.npy','time.npy','house_demand.npy','ev_demand.npy','pv_generation.npy']
dir = ('data/')

def statistics(dataset):
    """takes in an np.array 1D and returns the mean and std of the dataset"""
    mean = np.mean(dataset)
    std = np.std(dataset)
    return mean, std

class MgSimulate(object):
    def __init__(self):
        self.LAMBDA = 0.001
        self.house_increase_factor = 4
        self.charge_options = np.arange(-100,100,5)
        self.max_charge = 480
        self.sell      = .07
        self.cost      = (np.array([  .13,.13,.13,.13,.13,.13,.13,.13,.16,.16,.16,.16,.16,
                                    .16,.22,.22,.22,.22,.22,.16,.16,.16,.13,.13]))
        self.min_storage = np.array ([240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,200,240])

    def load_data(self,dir,data_file):
        '''
        Input:
            list (simulation info, datetime, house demand, ev demand, pv generation)
        example:
        data_file = ['simulation.npy','time.npy','house_demand.npy','ev_demand.npy','pv_generation.npy']

        where:
        simulation.npy     ->   contains a numeric representation that with unique values
                            that represent the season where the simulation belongs to
        time.npy           ->   contains a datetime code representing the specific moment
                            of the day where the data belongs to
        house_demand.npy   ->   contains the house demand in kwh
        ev_demand.npy      ->   contains the ev demand in kwh
        pv_generation.npy  ->   contains the pv generation in kwh

        '''
        self.id = np.load(dir+data_file[0])
        self.time = np.load(dir+data_file[1])
        self.house_dem = np.load(dir+data_file[2])
        self.ev_dem = np.load(dir+data_file[3])
        self.pv_gen = np.load(dir+data_file[4])

    def get_general_statistics(self):
        """
        objective - this function contains the probabilistic statistics for the transition
        of the environment. This will be used by latter functions to compute transitions
        from states.

        input  -

        output -
        sufixes
        me - mean,std """
        #TODO change the seasons to obtain different results and see how the algo
        #adapts to changes in the parameters

        n,d = 24,2
        self.house_dem_me = np.zeros([n,d])
        self.ev_dem_me = np.zeros([n,d])
        self.pv_gen_me = np.zeros([n,d])
        self.environmente_dist = {}
        for i in range(24):
            self.environmente_dist[i] = np.zeros(6)

        for i in range(24):
            """
            repeted code below, I realized late in the game that I wanted to work with
            dictionaries and not numpy arrays
            """
            #TODO fix the repeted code below

            indices = np.where(self.time==self.time[i])
            self.house_dem_me[i,0],self.house_dem_me[i,1] = statistics(np.take(self.house_dem,indices))
            self.environmente_dist[i,0],self.environmente_dist[i,1] = statistics(np.take(self.house_dem,indices))

            self.ev_dem_me[i,0],self.ev_dem_me[i,1] = statistics(np.take(self.ev_dem,indices))
            self.environmente_dist[i,2],self.environmente_dist[i,3] = statistics(np.take(self.ev_dem,indices))

            self.pv_gen_me[i,0],self.pv_gen_me[i,1] = statistics(np.take(self.pv_gen,indices))
            self.environmente_dist[i,4],self.environmente_dist[i,5] = statistics(np.take(self.pv_gen,indices))

    def generate_rollouts(self,num_rollouts):
        """
        objective -  use the statistics computed above to run the environment and obtain
        rollouts of the system.

        input - int. number of rollouts that need to be computed starting at 12pm and ending
        at 11am

        sufixes
        s - rollout. Array of length time_horizon*num_rollouts """
        self.period_end = np.zeros((num_rollouts,24))
        self.house_dem_s = np.zeros((num_rollouts,24))
        self.ev_dem_s = np.zeros((num_rollouts,24))
        self.pv_gen_s = np.zeros((num_rollouts,24))
        self.en_cos_s = np.zeros((num_rollouts,24))
        self.bat_lim_s = np.zeros((num_rollouts,24))

        self.period_end[:,23] = 1

        for i in range(24):
            self.house_dem_s[:,i] = np.random.normal(self.house_dem_me[i,0]*self.house_increase_factor,self.house_dem_me[i,1],num_rollouts)
            # self.ev_dem_s[:,i] = np.random.normal(self.ev_dem_me[i,0],self.ev_dem_me[i,1],num_rollouts)
            self.pv_gen_s[:,i] = np.random.normal(self.pv_gen_me[i,0],self.pv_gen_me[i,1]/10,num_rollouts)
            self.en_cos_s[:,i] = self.cost[i]
            self.bat_lim_s[:,i] = self.min_storage[i]

        self.house_dem_s[self.house_dem_s<0] = 0
        self.ev_dem_s[self.ev_dem_s<0] = 0
        self.pv_gen_s[self.pv_gen_s<0] = 0
        return (self.period_end,self.house_dem_s,self.ev_dem_s,self.pv_gen_s,self.en_cos_s,self.bat_lim_s)



    def get_action_space(self):
        """
        objective - return the action space of the algorithm. In this case the
                    possible charging or discharginf of the battery

        """
        return(self.charge_options,self.max_charge)

    def get_observation_space(self):
        """
        objective - return the observation space of the algorithm. We assume the
        markov property holds and the following states are a function of the current
        state
        """
        return(6)

    def generate_state(self,time):
        """
        objective -  given a time of interest, it will randomly generate a state

        input - int representing the time in question
        """
        period_end = 0
        if time > 23:
            time = 0
        if time ==23:
            period_end=1
        house_dem = np.random.normal(self.house_dem_me[time,0]*self.house_increase_factor,self.house_dem_me[time,1],1)
        pv_gen = np.random.normal(self.pv_gen_me[time,0],self.pv_gen_me[time,1]/10,1)
        ev_dem = 0
        en_cos = self.cost[time]
        return(house_dem[0],pv_gen[0],ev_dem,en_cos,period_end,time)


    def simulation_start_settup(self):
        house_dem,pv_gen,ev_dem,en_cos,period_end,step_time = self.generate_state(0)
        initial_state= {"time" : [[],[step_time]],
                "period_end" : [[],[period_end]],
                "house_dem" : [[],[house_dem]],
                "ev_dem" : [[],[ev_dem]],
                "pv_gen": [[],[pv_gen]],
                "en_cos": [[],[en_cos]],
                "bat_charge": [[],[self.min_storage[0]]],
                "action":[[],[]],
                "reward":[[],[]]}
        return initial_state

    def simulation_start(self):
        self.state = self.simulation_start_settup()

    def get_rewards(self):
        energy_balance = 0
        reward = 0
        if self.state["action"][-1][-1]>=0:
            energy_balance = self.state["pv_gen"][-1][-1] - (self.state["house_dem"][-1][-1] +
                                                            self.state["ev_dem"][-1][-1] +
                                                            self.state["action"][-1][-1])
        else:
            energy_balance = (self.state["pv_gen"][-1][-1] + self.state["action"][-1][-1] -
                             (self.state["house_dem"][-1][-1] + self.state["ev_dem"][-1][-1]))

        if self.state["bat_charge"][-1][-1] <= 30:
            reward = -1000
        elif self.state["bat_charge"][-1][-1] >= 450:
            reward = -1000
        else:
            if energy_balance >=0:
                reward = self.sell*energy_balance
            else:
                reward = self.cost[self.state["time"][-1][-1]]*energy_balance
        return(reward)

    def step_environment(self,action):
        """
        objective - given a state and an action, computes the next state and
        the rewards
        """
        house_dem,pv_gen,ev_dem,en_cos,period_end,step_time = self.generate_state(self.state["time"][-1][-1]+1)

        if step_time == 0:
            self.state["time"].append([step_time])
            self.state["period_end"].append([period_end])
            self.state["house_dem"].append([house_dem])
            self.state["ev_dem"].append([ev_dem])
            self.state["pv_gen"].append([pv_gen])
            self.state["en_cos"].append([en_cos])
            # self.state["bat_charge"].append([self.state["bat_charge"][-1][-1]+action])
            self.state["bat_charge"].append([self.min_storage[0]])
            self.state["action"].append([])
            self.state["reward"].append([])
        else:
            self.state["time"][-1].append(step_time)
            self.state["period_end"][-1].append(period_end)
            self.state["house_dem"][-1].append(house_dem)
            self.state["ev_dem"][-1].append(ev_dem)
            self.state["pv_gen"][-1].append(pv_gen)
            self.state["en_cos"][-1].append(en_cos)
            self.state["bat_charge"][-1].append(self.state["bat_charge"][-1][-1]+action)
            self.state["action"][-1].append(action)
            reward = self.get_rewards()
            self.state["reward"][-1].append(reward)

    def get_database(self):
        return(self.state)
