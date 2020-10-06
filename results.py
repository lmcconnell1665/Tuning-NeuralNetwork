"""
Tuning-NeuralNetwork-ResultsGeneration
10/4/2020
Dunn, McGinnis, McConnell
"""

import json
from matplotlib import pyplot as plt
import numpy as np

# read in data
with open('outputfile.txt') as f:
    json_data = json.load(f)

# number of iterations completed
len(json_data)

# find the lowest loss
seq = [x['loss'] for x in json_data]
next(item for item in json_data if item["loss"] == min(seq))

# Batch size vs time plot
plot_x =  [ sub['time'] for sub in json_data ] 
plot_y = [ sub['batch_size'] for sub in json_data ] 

plt.title("Plot") 
plt.xlabel("Time") 
plt.ylabel("Batch Size") 
plt.plot(plot_x,plot_y) 
plt.show()

# Boxplot SGD vs Adam loss
SGD_list = list()
Adam_list = list()

for sub in json_data:
    if sub['optimizer'] == 'SGD':
        SGD_list.append(sub)
    else:
        Adam_list.append(sub)

plot_x_SGD = [ sub['loss'] for sub in SGD_list ]
plot_x_Adam = [ sub['loss'] for sub in Adam_list ]

plot_keep_x_SGD = list()
plot_keep_x_Adam = list()

for i in range(len(plot_x_SGD)):
    if np.isnan(plot_x_SGD[i]) == False:
        plot_keep_x_SGD.append(plot_x_SGD[i])
        
for i in range(len(plot_x_Adam)):
    if np.isnan(plot_x_Adam[i]) == False:
        plot_keep_x_Adam.append(plot_x_Adam[i])

plt.boxplot(plot_keep_x_SGD)
plt.boxplot(plot_keep_x_Adam)
