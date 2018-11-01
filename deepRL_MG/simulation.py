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
    def _init_(self):
        self.LAMBDA = 0.001

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
        self.cost      = np.array([  .13,.13,.13,.13,.13,.13,.13,.13,.16,.16,.16,.16,.16,
                                    .16,.22,.22,.22,.22,.22,.16,.16,.16,.13,.13])
        self.min_storage = np.array ([240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.id = np.load(dir+data_file[0])
        self.time = np.load(dir+data_file[1])
        self.house_dem = np.load(dir+data_file[2])
        self.ev_dem = np.load(dir+data_file[3])
        self.pv_gen = np.load(dir+data_file[4])

    def get_general_statistics(self):
        """ sufixes
        me - mean,std """
        n,d = 24,2
        self.house_dem_me = np.zeros([n,d])
        self.ev_dem_me = np.zeros([n,d])
        self.pv_gen_me = np.zeros([n,d])
        for i in range(24):
            indices = np.where(self.time==self.time[i])
            self.house_dem_me[i,0],self.house_dem_me[i,1] = statistics(np.take(self.house_dem,indices))
            self.ev_dem_me[i,0],self.ev_dem_me[i,1] = statistics(np.take(self.ev_dem,indices))
            self.pv_gen_me[i,0],self.pv_gen_me[i,1] = statistics(np.take(self.pv_gen,indices))

    def generate_rollouts(self,num_rollouts):
        """ sufixes
        s - rollout. Array of length time_horizon*num_rollouts """
        self.period_end = np.zeros((num_rollouts,24))
        self.house_dem_s = np.zeros((num_rollouts,24))
        self.ev_dem_s = np.zeros((num_rollouts,24))
        self.pv_gen_s = np.zeros((num_rollouts,24))
        self.en_cos_s = np.zeros((num_rollouts,24))
        self.bat_lim_s = np.zeros((num_rollouts,24))

        for i in range(num_rollouts):
            if i == num_rollouts-1:
                self.period_end[:,i] = 1
            else:
                self.period_end[:,i] = 0
        for i in range(24):
            self.house_dem_s[:,i] = np.random.normal(self.house_dem_me[i,0],self.house_dem_me[i,1],num_rollouts)
            self.ev_dem_s[:,i] = np.random.normal(self.ev_dem_me[i,0],self.ev_dem_me[i,1],num_rollouts)
            self.pv_gen_s[:,i] = np.random.normal(self.pv_gen_me[i,0],self.pv_gen_me[i,1],num_rollouts)
            self.en_cos_s[:,i] = self.cost[i]
            self.bat_lim_s[:,i] = self.min_storage[i]

        self.house_dem_s[self.house_dem_s<0] = 0
        self.ev_dem_s[self.ev_dem_s<0] = 0
        self.pv_gen_s[self.pv_gen_s<0] = 0
        return (self.period_end,self.house_dem_s,self.ev_dem_s,self.pv_gen_s,self.en_cos_s,self.bat_lim_s)


if __name__ == "__main__":
    simulation = MgSimulate()
    simulation.load_data(data_complete)
    simulation.get_general_statistics()
    period_end,house_dem_s,ev_dem_s,pv_gen_s,en_cos_s,bat_lim_s = simulation.generate_rollouts(20)
