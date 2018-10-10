import pickle
import numpy as np
import matplotlib.pyplot as plt

#Load the data from the folder
#to plot part 1
# with open('fed1faed-5579-4b6c-98d3-22d89dac9da4.pkl','rb') as f:
#     data=pickle.load(f)
#to plot part 2
#data for double_q = False
with open('451e4dd0-02a5-4891-92a7-536c13061e9e.pkl','rb') as f:
    data=pickle.load(f)

# with open('668775b2-9250-4096-a2d1-923012c16b6d.pkl','rb') as f:
#     data=pickle.load(f)
n = len(data)
y = np.zeros(int(n/100))
mean_ = np.zeros(int(n/100))
fs = 12
data_max = np.zeros(n)
count = 0
for i in range(int(n/100)):
    y[i] = count
    count += 180000
    mean_[i] = np.mean(data[i*100:i*100+100])

count2 = 0
y_2 = np.zeros(n)
data_max = np.zeros(n)
for i in range(n):
    data_max[i] = np.amax(data[:i+1])
    y_2[i] = count2
    count2 += 1800

plt.plot(y,mean_,label="AverageReturn")
plt.plot(y_2,data_max,label="MaxAverageReturn")
plt.xlabel('Timestep', fontsize = (fs))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('AverageReturn', fontsize = (fs))
name_for_graphs = ('Question 2: Q-learning performance.')
plt.title(name_for_graphs, fontsize= (fs))
plt.show()
