"""
originally developed by Ramon Crespo Fall 2018 for CS 294-112 final project
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import inspect
from multiprocessing import Process
import pdb
import expert as exp

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass)

        Hint: use tf.layers.dense
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for i in range(n_layers):
            output_placeholder = tf.layers.dense(inputs=output_placeholder, units=size, activation=activation)
        output_placeholder = tf.layers.dense(inputs=output_placeholder, units=output_size, activation=output_activation)
    return output_placeholder


#============================================================================================#
# Actor Critic
#============================================================================================#

class Agent(object):
    def __init__(self):
        self.lamb = .99
        self.horizon = 24
        self.ob_dim = self.horizon*6
        self.ac_dim = 24
        self.n_layers = 2
        self.size = 64
        self.learning_rate = 1

    def expert_data(self):
        """
        Objective -
                Take advantage of previous rollouts from the expert to train the
                initial nn. This saves time as there is no need to generate new
                rollouts from the expert
        input  -
                Make sure the specified files in the dictionary are available for
                the expert to read
        output -
                Expert dictionary:
                0,1,2,3,4 - shape(num_rollouts,horizon)
        """
        self.expert_ob =   {0:np.load("simulation/period_end.npy"),
                            1:np.load("simulation/house_dem_s.npy"),
                            2:np.load("simulation/ev_dem_s.npy"),
                            3:np.load("simulation/pv_gen_s.npy"),
                            4:np.load("simulation/en_cos_s.npy"),
                            5:np.load("simulation/bat_lim_s.npy")}
        self.expert_ac = np.load("simulation/e.npy")

    def expert_sample(self,batch_size):
        """
        Objective -
                Everytime this function is called it will randomly sample values
                from the self.expert data and assign this values to a dictionary
                of size batch size
        input  -
                batch_size (int)
        output -
                --
        """
        n,d = self.expert_ob[1].shape
        sample = np.random.randint(0,n,50)
        expert_sample_ob = {}
        expert_sample_ac = []
        for i in range(6):
            expert_sample_ob[i] = np.zeros((batch_size,self.horizon))

        for i in range(batch_size):
            expert_sample_ac.append(self.expert_ac[sample[i],:])
            for t in range(6):
                expert_sample_ob[t][i,:]=self.expert_ob[t][sample[i],:]

        return(expert_sample_ob,expert_sample_ac)


    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions
            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
        """
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.int32)
        return sy_ob_no, sy_ac_na


    def train_agent(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            actor - tries to mimic the behavior of the expert
        """
        # define placeholder
        self.sy_ob_no, self.sy_ac_na = self.define_placeholders()

        #build nn
        self.actor_strategy = tf.squeeze(build_mlp(
                                self.sy_ob_no,
                                24,
                                "nn_actor",
                                n_layers=self.n_layers,
                                size=self.size))

        #loss function
        actor_loss = tf.losses.absolute_difference(self.actor_strategy,self.sy_ac_na)
        #train step
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)
        print()


    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self,rollouts=1):
        expert = exp.Expert()
        period_end,house_dem,ev_dem,pv_gen,en_cos,bat_lim = expert._dataset(rollouts)
        path = {"period_end" : np.array(period_end, dtype=np.float32),
                "house_dem" : np.array(house_dem, dtype=np.float32),
                "ev_dem" : np.array(ev_dem, dtype=np.float32),
                "pv_gen": np.array(pv_gen, dtype=np.float32),
                "en_cos": np.array(en_cos, dtype=np.float32),
                "bat_lim": np.array(bat_lim, dtype=np.float32)}
        return path

    def actor_action(self, path):
        """
            Generates an action given an observation.

            arguments:
                path: dictionary return from sample_trajectory

            returns:
                e : the SOC of the battery at each timestep
        """
        ob_no = np.concatenate((path["period_end"],path["house_dem"],
                                path["ev_dem"],path["pv_gen"],path["en_cos"],
                                path["bat_lim"]),axis=1)
        e = self.sess.run(self.actor_strategy, feed_dict={self.sy_ob_no: ob_no})
        return e


    def update_actor(self, ob_no, ac_na):
        """
            Update the parameters of the policy.

            arguments:
                ob_no: shape: dictionary
                ac_na: shape: (sum_of_path_lengths).

            returns:
                nothing

        """
        observation_no = np.concatenate((ob_no[0],ob_no[1],
                                ob_no[2],ob_no[3],ob_no[4],
                                ob_no[5]),axis=1)
        self.sess.run(self.actor_update_op,
            feed_dict={self.sy_ob_no: observation_no, self.sy_ac_na: ac_na})

def main():
    batch_size = 50
    agent = Agent()
    agent.expert_data()
    agent.train_agent()
    agent.init_tf_sess()
    for i in range(1000):
        expert_sample_ob,expert_sample_ac = agent.expert_sample(batch_size)
        agent.update_actor(expert_sample_ob,expert_sample_ac)
    sample_path = agent.sample_trajectory()
    soc = agent.actor_action(sample_path)
    print(soc)

if __name__ == "__main__":
    main()
