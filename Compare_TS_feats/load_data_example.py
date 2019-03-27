# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import h5py
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from scipy.io import loadmat
from scipy.ndimage.filters import maximum_filter1d,median_filter
from scipy.ndimage.measurements import label
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier  

wdir            = r"C:\Users\Eamonn\Documents\GitHub\RC_utilities\Compare_TS_feats\example_feat_data"
fname_EtOH = r'feature_struct.mat'
fname_Ace = r'feature_struct2.mat'

os.chdir(wdir)

data = sio.loadmat(fname_EtOH)
EtOH_data = data['feature_struct']
EtOH_labels = EtOH_data['labels'][0,0]
EtOH_mat = EtOH_data['lv_norm'][0,0]

data = sio.loadmat(fname_Ace)
Ace_data = data['feature_struct2']
Ace_labels = Ace_data['labels'][0,0]
Ace_mat = Ace_data['lv_norm'][0,0]

num_feats = np.shape(Ace_mat)[1]
num_traces = np.shape(Ace_mat)[0]
num_messages = len(np.unique(Ace_labels))

acc = np.zeros(num_messages)
accuracy_matrix = np.zeros((num_feats,num_messages))
for feat_num in range(num_feats):
    
    #plt.plot(Ace_labels,Ace_mat[:,feat_num],'.')
    #plt.plot(EtOH_labels,EtOH_mat[:,feat_num],'.')
    #plt.xlabel('message ID')
    #plt.ylabel('Feature values')
    #plt.show
    
    for trace_num in range(num_messages):
    
        v1 = Ace_mat[np.where(Ace_labels==trace_num+1)[0],feat_num]
        v2 = EtOH_mat[np.where(EtOH_labels==trace_num+1)[0],feat_num]
        num_obs = len(v1)+len(v2)
        vals_feat = (np.concatenate((v1,v2),0))
        label_feat = np.concatenate((np.zeros(len(v1)),np.ones(len(v2))),0)
        
        # Iterate 1NN over all points and test against the true labels
        label_1nn = np.zeros(num_obs)
        for i in range(num_obs):
            distances = np.abs(vals_feat - vals_feat[i])
            distances[i] = np.max(distances) + 1 # rule out the value itself
            label_1nn[i] = label_feat[np.where(distances==np.min(distances))[0][0]]
#         print(trace_num,np.sum(1*(label_feat==label_1nn))/num_obs)    
        acc[trace_num] = np.sum(1*(label_feat==label_1nn))/num_obs
        
    accuracy_matrix[feat_num,:] = acc
    
plt.stem(np.mean(accuracy_matrix,1))
plt.show()
plt.stem(np.mean(accuracy_matrix,0))
plt.show()  
     
#plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat)
#plt.plot(label_1nn)
#classifier = KNeighborsClassifier(n_neighbors=1)  
#classifier.fit(vals_feat_subset, label_feat_subset) 



