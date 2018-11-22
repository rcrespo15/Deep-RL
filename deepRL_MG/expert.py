from numpy import linalg as LA
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd
import itertools
import sqlite3
import time
import os
import csv
import simulation as sim

def give_negative(values):
    empty_list = []
    for i in range(0,len(values),1):
        empty_list.append(values[i]*-1)
    return empty_list

def graph(b_d,b_c,G_E,E,Z,G_I,pv_generation,house_demand,cost_to_buy):
    fs = 12
    simulation_horizon = len(b_d)
    dates = [1493600340 + 3600*i for i in range(simulation_horizon)]
    labels = [time.strftime('%H:%M:%S', time.gmtime(date)) for date in dates]

    to_plot = [(G_I, "Grid Import [kW]"),(G_E, "Grid Export [kW]"),
               (give_negative(b_d), "Flywheel Discharge [kWh]"),(house_demand, 'load demand [kWh]'),
               (Z, "Power Delivered to EVSE [kWh]"),(pv_generation, "PV Generation [kWh]"),
               (cost_to_buy*1000, 'price to buy [$/MWh]'),(b_c, "Flywheel Charge [kWh]"),
              (E,'SOC [kWh]')]
    plt.figure(figsize=(20, 10))
    for i in range(len(to_plot)):
        plt.plot(dates, to_plot[i][0],label=to_plot[i][1])
        plt.xticks(dates, labels, rotation='vertical')
        plt.tick_params(labelsize=10)
        plt.xlabel('Time of Day', fontsize = (fs + 4))
    name_for_graphs = ('Simulation')
    plt.title(name_for_graphs, fontsize= (fs + 4))
    plt.grid()
    plt.legend( fontsize = (fs + 4),loc=2)
    plt.show()

class Expert(object):
    def _init_(self):
        self.LAMBDA = 0.001

    def _dataset(self,rollouts):
        self.data_complete = [  'simulation.npy','time.npy','house_demand.npy',
                                'ev_demand.npy','pv_generation.npy']
        self.dir = ('data/')
        self.simulation = sim.MgSimulate()
        self.simulation.load_data(self.dir,self.data_complete)
        self.simulation.get_general_statistics()
        period_end,house_dem_s,ev_dem_s,pv_gen_s,en_cos_s,bat_lim_s = self.simulation.generate_rollouts(rollouts)
        return(period_end,house_dem_s,ev_dem_s,pv_gen_s,en_cos_s,bat_lim_s)

    def _optimize(self,pv_generation,ev_demand,house_demand,costdata,storage):
        """
        This tool evaluates the optimal strategy for a 24 hour period.
        input:
            pv_generation
            ev_demand
            house_demand
            costdata
            storage
        output:
            b_d = Power discharged from flywheel
            b_c = Power delivered to flywheel
            G_E = Power exported to the grid
            G_I = Power imported from grid
            C = Max charging capacity of EV
            E = Energy Level of Flywheel
            Z = Cumulative energy delivered to EV

        """
        costsell=0.07
        E_max=480
        b_max=100
        C_max= (10*24)
        chargeff=0.9
        diseff=0.9
        chargeff_ev=0.9
        delta_t = 1
        num_variables = len(pv_generation)

        b_d = Variable(num_variables)
        b_c = Variable(num_variables)
        G_E = Variable(num_variables)
        C   = Variable(num_variables)
        E   = Variable(num_variables)
        Z   = Variable(num_variables)
        G_I = Variable(num_variables)

        objective = Minimize((delta_t)*(costdata.T*G_I) - sum(costsell*G_E)*(delta_t))

        constraints = [ G_I == house_demand + b_c + C + G_E - pv_generation - b_d]
        constraints += [E[0] == storage[0]]
        constraints += [ Z[0] == ev_demand[0]]

        #state boundary conditions at time = 24
        constraints += [G_E[num_variables-1] >= 0]
        constraints += [G_I[num_variables-1] >= 0]


        constraints += [b_c[num_variables-1] >= 0]
        constraints += [b_c[num_variables-1] <= b_max]
        constraints += [b_d[num_variables-1] >= 0]
        constraints += [b_d[num_variables-1] <= b_max]

        for k in range(0,num_variables-1,1):
            constraints += [E[k+1] == E[k] + (chargeff*b_c[k] - (1/diseff)*b_d[k])*delta_t]
            constraints += [E[k]   >= 0]
            constraints += [E[k+1] >= storage[k+1]]
            constraints += [G_E[k] >= 0]
            constraints += [G_I[k] >= 0]
            constraints += [E[k]   <= E_max]
            constraints += [b_c[k] >= 0]
            constraints += [b_c[k] <= b_max]
            constraints += [b_d[k] >= 0]
            constraints += [b_d[k] <= b_max]

            # EV charging dynamics
            constraints += [Z[k+1] == Z[k] + (chargeff_ev*C[k]*delta_t)]
    #         constraints += [Z[k] >= sum(ev_demand[:k])]
            constraints += [C[k] >= 0]
            constraints += [C[k] <= C_max]
            constraints += [Z[k+1] >= .95*sum(ev_demand[:(k+2)])]
            constraints += [Z[k+1] <= 1.05*sum(ev_demand[:(k+2)])]
            constraints += [C[k+1] >= 0]
            constraints += [C[k+1] <= C_max]

        prob = Problem(objective, constraints)
        prob.solve()

        b_d_answer = []
        b_c_answer = []
        G_E_answer = []
        C_answer = []
        E_answer = []
        Z_answer = []
        G_I_answer = []

        for i in range(0,num_variables,1):
            b_d_answer.append(b_d[i].value)
            b_c_answer.append(b_c[i].value)
            G_E_answer.append(G_E[i].value)
            C_answer.append(C[i].value)
            E_answer.append(E[i].value)
            Z_answer.append(Z[i].value)
            G_I_answer.append(G_I[i].value)

        return (b_d_answer,b_c_answer,G_E_answer,C_answer,E_answer,Z_answer,G_I_answer,prob.value)

    def computational_graph(self,rollouts):
        self.period_end,self.house_dem_s,self.ev_dem_s,self.pv_gen_s,self.en_cos_s,self.bat_lim_s = self._dataset(rollouts)
        self.b_d = np.zeros((rollouts,24))
        self.b_c = np.zeros((rollouts,24))
        self.g_e = np.zeros((rollouts,24))
        self.c = np.zeros((rollouts,24))
        self.e = np.zeros((rollouts,24))
        self.z = np.zeros((rollouts,24))
        self.g_i = np.zeros((rollouts,24))
        self.ret = np.zeros(rollouts)
        for i in range(rollouts):
            self.b_d[i,:],self.b_c[i,:],self.g_e[i,:],self.c[i,:],self.e[i,:], self.z[i,:],self.g_i[i,:],self. ret[i] =    self._optimize(self.pv_gen_s[i,:],
                                                        self.ev_dem_s[i,:],self.house_dem_s[i,:],
                                                        self.en_cos_s[i,:],self.bat_lim_s[i,:])
        self.hour_ret = self.g_i*self.en_cos_s+self.g_e*.07
        return(self.b_d,self.b_c,self.g_e,self.c,self.e,self.z,self.g_i,self.hour_ret)

    # def give_negative(self,values):
    #     empty_list = []
    #     for i in range(0,len(values),1):
    #         empty_list.append(values[i]*-1)
    #     return empty_list

    # def graph_results(self):
    #     fs = 12
    #     simulation_horizon = 24 #self.b_d.shape
    #     dates = [1493600340 + 3600*i for i in range(simulation_horizon)]
    #     labels = [time.strftime('%H:%M:%S', time.gmtime(date)) for date in dates]
    #
    #     to_plot = [(self.g_i.T, "Grid Import [kW]"),(self.g_e.T, "Grid Export [kW]"),
    #                (self.give_negative(self.b_d).T, "Flywheel Discharge [kWh]"),(self.house_dem_s.T, 'load demand [kWh]'),
    #                (self.z.T, "Power Delivered to EVSE [kWh]"),(self.pv_gen_s.T, "PV Generation [kWh]"),
    #                (self.en_cos_s.T*1000, 'price to buy [$/MWh]'),(self.b_c.T, "Flywheel Charge [kWh]"),
    #               (self.e.T,'SOC [kWh]')]
    #     plt.figure(figsize=(20, 10))
    #     for i in range(len(to_plot)):
    #         plt.plot(dates, to_plot[i][0],label=to_plot[i][1])
    #         plt.xticks(dates, labels, rotation='vertical')
    #         plt.tick_params(labelsize=10)
    #         plt.xlabel('Time of Day', fontsize = (fs + 4))
    #     name_for_graphs = ('Simulation')
    #     plt.title(name_for_graphs, fontsize= (fs + 4))
    #     plt.grid()
    #     plt.legend( fontsize = (fs + 4),loc=2)
    #     plt.show()

# 
# if __name__ == "__main__":
#     rollouts = 5000
#     expert = Expert()
#     period_end,house_dem_s,ev_dem_s,pv_gen_s,en_cos_s,bat_lim_s = expert._dataset(rollouts)
    # b_d,b_c,g_e,c,e,z,g_i,hour_ret = expert.computational_graph(rollouts)
    # np.save("simulation/bat_lim_s", bat_lim_s)
    # np.save("simulation/house_dem_s", house_dem_s)
    # np.save("simulation/ev_dem_s", ev_dem_s)
    # np.save("simulation/pv_gen_s", pv_gen_s)
    # np.save("simulation/en_cos_s", en_cos_s)
    # np.save("simulation/b_d", b_d)
    # np.save("simulation/b_c", b_c)
    # np.save("simulation/g_e", g_e)
    # np.save("simulation/c", c)
    # np.save("simulation/e", e)
    # np.save("simulation/z", z)
    # np.save("simulation/g_i", g_i)
    # np.save("simulation/hour_ret", hour_ret)
    #
    # graph(b_d.T,b_c.T,g_e.T,e.T,z.T,g_i.T,pv_gen_s.T,house_dem_s.T,en_cos_s.T)
