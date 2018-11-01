import pickle
import numpy as np
import matplotlib.pyplot as plt

#Load the data from the folder
#to plot part 1
# with open('fed1faed-5579-4b6c-98d3-22d89dac9da4.pkl','rb') as f:
#     data=pickle.load(f)
#to plot part 2
#data for double_q = False
# with open('451e4dd0-02a5-4891-92a7-536c13061e9e.pkl','rb') as f:
#     data=pickle.load(f)

#data for double_q = True
#'6d08e906-154f-4009-a150-76f389331ca4.pkl'

#to plot part 3
#data for epsiolon =5
#053d41af-f190-4aef-a4cc-a5949ae3e45a.pkl
#data for epsiolon =.5
#2d5f4273-f0fb-44d6-8d29-3c5af1e403a8.pkl
#data for epsiolon =.001
#296df43e-3e0a-4e39-b684-a0d4a56214a9.pkl

fs = 12

def clean_data(filename):
    with open(filename,'rb') as f:
        data=pickle.load(f)

    n = len(data)
    timesteps = 500000
    timesteps_per_reading = timesteps/n
    y = np.zeros(int(n/100))
    mean_ = np.zeros(int(n/100))
    count = 0
    for i in range(int(n/100)):
        y[i] = count
        count += timesteps_per_reading
        mean_[i] = np.mean(data[i*100:i*100+100])
        count += timesteps_per_reading*100
    return (y,mean_)

def plot_data(y,mean1,y2,mean2,y3,mean3,y4,mean4,title):
    plt.plot(y,mean1,label="AverageReturn - epsilon = .1")
    plt.plot(y2,mean2,label="AverageReturn - epsilon = .001")
    plt.plot(y3,mean3,label="AverageReturn - epsilon = .5")
    plt.plot(y4,mean4,label="AverageReturn - epsilon = 5")
    plt.xlabel('Timestep', fontsize = (fs))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('AverageReturn', fontsize = (fs))
    name_for_graphs = (title)
    plt.title(name_for_graphs, fontsize= (fs))
    plt.show()

y_dqr_1,mean_dqr_1 = clean_data('451e4dd0-02a5-4891-92a7-536c13061e9e.pkl')
y_dqr_001,mean_dqr_001 = clean_data('6d08e906-154f-4009-a150-76f389331ca4.pkl')
y_dqr_5,mean_dqr_5 = clean_data('2d5f4273-f0fb-44d6-8d29-3c5af1e403a8.pkl')
y_dqr_50,mean_dqr_50 = clean_data('053d41af-f190-4aef-a4cc-a5949ae3e45a.pkl')

# y_dqr_double,mean_dqr_double = clean_data('6d08e906-154f-4009-a150-76f389331ca4.pkl')
plot_data(  y_dqr_1,mean_dqr_1,y_dqr_001,mean_dqr_001,
            y_dqr_5,mean_dqr_5,y_dqr_50,mean_dqr_50,"Results for different values of epsilon - LunarLander")
