# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5** older version -- might need to do some changes
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.

#################################################################################

To run the code:
Part 1 ->
Question 1 -> just follow the instructions in the hw without making any changes to the code
Question 2 -> use run_dqn_lander.py and change line 35 in dqn.py from True to False depending
if you want doble DQN or regular DQN
Question 3 -> use run_dqn_lander.py and change line 24 from .1 to .001 to .5 to 5

to plot results go to plot_part_1.py and call the functions with the correct code. plot_part_1.py
already contains the files that were plotted to produce the results

Part2 ->
Question 1 -> follow instructions in homeworks
Question 2 -> follow instructions in homeworks and insert 100 and 100 for the -ntu and -ngsptu tags
use the plot function to plot the results
