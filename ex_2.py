# -*- coding: utf-8 -*-
"""
Created on Sat May  10 15:22:10 2019

@author: foysal
"""

"""I have run the same program from ex1 and just saved the timing here and plotted the graph here"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
x = np.array([1, 2, 3, 4, 5])

speed_up=[]
parallel_effeciency = []
# of process = 1
serial_process_execution = 4967.93
total_parallel_execution = 4967.93

speed_up.append(serial_process_execution/total_parallel_execution)
sp=serial_process_execution/total_parallel_execution
parallel_effeciency.append(sp/1)
# of process  = 2

serial_process_execution_2 = 4967.93
total_parallel_execution_2 = 3227.60
speed_up.append(serial_process_execution_2/total_parallel_execution_2)
sp_2=serial_process_execution_2/total_parallel_execution_2
parallel_effeciency.append(sp_2/2)

# n of w = 3
serial_process_execution_3 = 4967.93
total_parallel_execution_3 = 3265.47
speed_up.append(serial_process_execution_3/total_parallel_execution_3)
sp_3=serial_process_execution_3/total_parallel_execution_3
parallel_effeciency.append(sp_3/3)


# n of w = 4
serial_process_execution_4 = 4967.93
total_parallel_execution_4 = 2485.42
speed_up.append(serial_process_execution_4/total_parallel_execution_4)
sp_4=serial_process_execution_4/total_parallel_execution_4
parallel_effeciency.append(sp_4/4)


# of w = 5

serial_process_execution_5 = 4967.93
total_parallel_execution_5 = 2183.16
speed_up.append(serial_process_execution_5/total_parallel_execution_5)
sp_5=serial_process_execution_5/total_parallel_execution_5
parallel_effeciency.append(sp_5/5)


##x_new = np.linspace(x.min(), x.max(), 200)

##SpeedUp Graph vs Number of Process
speed_up_array = np.array(speed_up)
print('speed_up: ',speed_up)
##y_new = spline(x, y, x_new)
plt.plot(x, speed_up, 'g', label = "Original Curve")

plt.xlabel("No of process")
plt.ylabel("Sp")
plt.title("Number of Process vs SpeedUp Graph for cluster size k =2")
plt.show()

##parallel efficiency vs Number of Process
parallel_effeciency_array = np.array(parallel_effeciency)
print('parallel_effeciency: ',parallel_effeciency)
##y_new = spline(x, y, x_new)
plt.plot(x, parallel_effeciency, 'g', label = "Original Curve")

plt.xlabel("No of process")
plt.ylabel("Parallel effeciency")
plt.title("Number of Process vs parallel efficiency for cluster size k =2")
plt.show()


