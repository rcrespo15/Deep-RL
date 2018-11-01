from numpy import linalg as LA
import numpy as np
import matplotlib as plt
from cvxpy import *
import pandas as pd
import itertools
import sqlite3
import time
import os
import csv
import simulation as sim


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
        The variables are:
            b_d = Power discharged from flywheel
            b_c = Power delivered to flywheel
            G_E = Power exported to the grid
            G_I = Power imported from grid
            C = Mac charging capacity of EV
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
        period_end,house_dem_s,ev_dem_s,pv_gen_s,en_cos_s,bat_lim_s = self._dataset(rollouts)
        b_d = np.zeros((rollouts,24))
        b_c = np.zeros((rollouts,24))
        g_e = np.zeros((rollouts,24))
        c = np.zeros((rollouts,24))
        e = np.zeros((rollouts,24))
        z = np.zeros((rollouts,24))
        g_i = np.zeros((rollouts,24))
        ret = np.zeros(rollouts)
        for i in range(rollouts):
            b_d[i,:],b_c[i,:],g_e[i,:],c[i,:],e[i,:],z[i,:],g_i[i,:],ret[i] = self._optimize(pv_gen_s[i,:],ev_dem_s[i,:],house_dem_s[i,:],en_cos_s[i,:],bat_lim_s[i,:])
        hour_ret = g_i*en_cos_s+g_e*.07
        return(b_d,b_c,g_e,c,e,z,g_i,hour_ret)


if __name__ == "__main__":
    expert = Expert()
    expert.computational_graph(24)
