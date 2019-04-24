# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:57:16 2019

@author: Eamonn
"""

import os

os.chdir(r'C:\Users\Eamonn\Documents\GitHub\RC_utilities\knndtw')

from knndtw import *

import knndtw

time = np.linspace(0,20,1000)
amplitude_a = 5*np.sin(time)
amplitude_b = 3*np.sin(time + 1)

m = KnnDtw()
distance = m._dtw_distance(amplitude_a, amplitude_b)

fig = plt.figure(figsize=(12,4))
_ = plt.plot(time, amplitude_a, label='A')
_ = plt.plot(time, amplitude_b, label='B')
_ = plt.title('DTW distance between A and B is %.2f' % distance)
_ = plt.ylabel('Amplitude')
_ = plt.xlabel('Time')
_ = plt.legend()