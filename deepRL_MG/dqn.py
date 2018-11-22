"""
originally developed by Ramon Crespo Fall 2018
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
import simulation as sim
from dqn_utils import *

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


def mg_model(state_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = state_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out
#============================================================================================#
# Actor Critic
#============================================================================================#

class QlearnAgent(object):
    def __init__(self):
        self.gama = .99
        self.horizon = 24
        self.n_layers = 2
        self.size = 64
        self.learning_rate = 1
        self.target_update_freq = 10
        self.batch_size = 50
        self.learning_freq = 4
        self.learning_starts = 5000
        self.env = sim.MgSimulate()
        self.ac_dim,_ = self.env.get_action_space()
        self.ob_dim = self.env.get_observation_space()
        self.double_q = False

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions
            returns:
                self.obs_t_ph: placeholder for observations
                self.act_t_ph: placeholder for actions
                self.rew_t_ph: placeholder for rewards
                self.obs_tp1_ph: placeholder for observation at t + 1
        """
        self.obs_t_ph = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.act_t_ph = = tf.placeholder(tf.int32,   [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(shape=[None, self.ob_dim], name="ob_t1", dtype=tf.float32)

    def computational_graph(self):
        """
        setup computational steps to compute the Bellman error
        """
        # self.define_placeholders()
        # self.Q_phi = mg_model(self.obs_t_ph,self.ac_dim, scope="q_func", reuse=False)
        # self.q_phi = tf.reduce_sum(tf.multiply(self.Q_phi, tf.one_hot(self.act_t_ph,depth=self.num_actions)),axis=1)

        self.Q_phi_p = q_func(self.obs_tp1_ph, self.ac_dim, scope="q_targ_func", reuse=False)
        self.q_phi_p_max = tf.reduce_max(self.Q_phi_p, axis=1)
        if self.done == 1:
           self.y_i = self.rew_t_ph
        else:
               self.y_i = self.rew_t_ph + self.gamma * self.q_phi_p_max

        return (self.y_i)
        # self.total_error = tf.reduce_mean(huber_loss(self.y_i-self.q_phi))
        # self.compute_grad = tf.train.AdamOptimizer(self.learning_rate).compute_gradients(self.total_error)

    def stopping_criterion_met(self):
        raise NotImplementedError

    def step_env(self):
        raise NotImplementedError

    def update_model(self):
        raise NotImplementedError

    def log_progress(self):
        raise NotImplementedError


def learn(*args, **kwargs):
    raise NotImplementedError
    # alg = QLearner(*args, **kwargs)
    # while not alg.stopping_criterion_met():
    #     alg.step_env()
    # # at this point, the environment should have been advanced one step (and
    # # reset if done was true), and self.last_obs should point to the new latest
    # # observation
    #     alg.update_model()
    #     alg.log_progress()




def main():
    data_complete = [  'simulation.npy','time.npy','house_demand.npy',
                            'ev_demand.npy','pv_generation.npy']
    dir = ('data/')
    env=sim.MgSimulate()
    env.load_data(dir,data_complete)
    env.get_general_statistics()
    action_space,_ = env.get_action_space()
    env.simulation_start()
    actions = np.random.randint(0,len(action_space),100)
    for i in actions:
        env.step_environment(action_space[i])
    data = env.get_database()


if __name__ == "__main__":
    main()
