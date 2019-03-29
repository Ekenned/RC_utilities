# -*- coding: utf-8 -*-
"""

Basic read in code for time series features
-Eamonn

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
import platform

#a = platform.uname()
#if a[1] == 'XPS15': # settings for graphs on 4K XPS15 only
#    plt.style.use(['seaborn-paper']) # ,'dark_background'])
#    matplotlib.rcParams['figure.figsize']   = (22, 22)
#    matplotlib.rcParams['axes.titlesize']   = 30
#    matplotlib.rcParams['axes.labelsize']   = 30
#    matplotlib.rcParams['lines.linewidth']  = 2
#    matplotlib.rcParams['lines.markersize'] = 20
#    matplotlib.rcParams['xtick.labelsize']  = 30
#    matplotlib.rcParams['ytick.labelsize']  = 30

#------------------------------------------------------------------------------
#%%

wdir = r"C:\Users\Eamonn\Documents\GitHub\RC_utilities\Compare_TS_feats\example_feat_data"
fname = r'chem_ts_feat'

# Load the data
os.chdir(wdir)
TSF_data = sio.loadmat(fname + '.mat')
label_dict = TSF_data[fname]['labels'][0][0]
feat_dict = TSF_data[fname]['feat_mat'][0,0]
chem_names = TSF_data[fname]['chem_names'][0,0]
num_chems = np.shape(chem_names)[0]
num_feats = np.shape(feat_dict[0][chem_names[0][0][0]][0])[0]

# Core information, pattern labels, features, chemicals for dictionaries
labels = {}
feat_mat = {}
chem_name = {}

# And some ancillary information
num_traces = np.zeros(num_chems)
feat_max_arr = np.zeros((num_feats,num_chems))
feat_min_arr = np.zeros((num_feats,num_chems))

# Get the labels and feature matrix for every chemical
for chem_num in range(num_chems):

    chem_name[chem_num] = chem_names[chem_num][0][0]
    feat_mat[chem_num] = feat_dict[chem_name[chem_num]][0,0]
    feat_mat[chem_num][np.isnan(feat_mat[chem_num])] = 0 # remove nan entries
    labels[chem_num] = label_dict[chem_name[chem_num]][0][0]
    num_traces[chem_num] = np.shape(feat_mat[chem_num])[1]
    feat_max_arr[:,chem_num] = np.max(feat_mat[chem_num],1)
    feat_min_arr[:,chem_num] = np.min(feat_mat[chem_num],1)
    
# Normalize data from 0 - 1 
abs_max = np.nanmax(feat_max_arr,1) # max and min over all chemicals by feature
abs_min = np.nanmin(feat_min_arr,1)
range_feat = abs_max - abs_min + 10**-6
feat_mat_norm = feat_mat
for chem_num in range(num_chems):
    
    for f in range(num_feats):
        all_trace_vals = feat_mat[chem_num][f,:]
        all_trace_vals = (all_trace_vals - abs_min[f]) / range_feat[f]
        feat_mat_norm[chem_num][f,:] = all_trace_vals    
        
#-----------------------------------------------------------------------------
#%%

#plt.loglog(np.median(Ace_mat,1),np.median(EtOH_mat,1),'.')
#plt.loglog(np.mean(Ace_mat,1),np.mean(EtOH_mat,1),'.')
#plt.loglog(np.max(Ace_mat,1),np.max(EtOH_mat,1),'.')
#plt.loglog(np.min(Ace_mat,1),np.min(EtOH_mat,1),'.')
#plt.xlim((10**-5,10**5));
#plt.ylim((10**-2,10**1))

num_msgs = len(np.unique(labels[c]))
acc = np.zeros(num_msgs)
accuracy_matrix = np.zeros((num_feats,num_msgs))
plot_feat_num = 2 # increase this to plot more graphs
for f in range(num_feats):
    
    if f<plot_feat_num:   
        for c in range(num_chems):        
            plt.scatter(labels[c],feat_mat_norm[c][f,:])   
        plt.title(f)
        plt.xlabel('message ID')
        plt.ylabel('Feature values')
        # plt.legend('EtOH','Ace')
        plt.show()
    
    # Seperate out data by every pattern sent,, determine chemical discrimination
    for trace_num in range(num_msgs):
        
        # Seperate out features by chemical, plot a 6x4 grid for 24 traces
        v1 = feat_mat_norm[0][f,np.where(labels[0]==trace_num+1)[0]]
        v2 = feat_mat_norm[1][f,np.where(labels[1]==trace_num+1)[0]]
        num_obs = len(v1)+len(v2)
        vals_feat = (np.concatenate((v1,v2),0))
        label_feat = np.concatenate((np.zeros(len(v1)),np.ones(len(v2))),0)
        if f<plot_feat_num: 
            plt.subplot(6,4,trace_num+1)
            plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat,vmin=-.5,vmax=1.5)
            plt.title(trace_num) 
            plt.ylim((0,1))
            plt.yticks([])
        
        # Iterate 1NN over all points and test against the true labels
        label_1nn = np.zeros(num_obs)
        for i in range(num_obs):
            distances = np.abs(vals_feat - vals_feat[i])
            distances[i] = np.max(distances) + 1 # rule out the value itself
            label_1nn[i] = label_feat[np.where(distances==np.min(distances))[0][0]]
        # print(trace_num,np.sum(1*(label_feat==label_1nn))/num_obs)    
        acc[trace_num] = np.sum(1*(label_feat==label_1nn))/num_obs
       
    plt.show()
        
    accuracy_matrix[f,:] = acc
     
num_sf = int(num_feats/num_vars)
plt.semilogy(1 - np.mean(accuracy_matrix[:,:],1),'.')
for i in range(7):
    plt.plot((i*num_sf,i*num_sf),(.0001,.3),'k')
plt.xlabel('Feature index, V = 1...219,P=220,...')
plt.ylabel('Error rate on chemical identification')
plt.xlim((0,num_feats))
plt.ylim((.001,.3))
plt.title('Single feature error identification rates averaged over all patterns')
plt.show() 

num_vars = 6
threshold = np.array([.9,.95,.98,.99,.995])
feats_found = np.zeros((len(threshold),num_vars))
tick = 0
for j in threshold:
    for i in range(num_vars):
        feats_found[tick,i] = np.sum(np.mean(accuracy_matrix[i*219:(i+1)*219,:],1)>=j)
    tick = tick+1

plt.plot(threshold,feats_found,'o')
plt.ylabel('# Features classifying with accuracy>X')
plt.xlabel('Accuracy')
plt.show() 

thresh = 0.98
inds_V = np.where(np.mean(accuracy_matrix[0:(1)*num_sf,:],1)>thresh )[0]
inds_T = np.where(np.mean(accuracy_matrix[2*num_sf:3*num_sf,:],1)>thresh )[0]
c_inds = np.intersect1d(inds_T,inds_V)

plt.stem(1 - np.mean(accuracy_matrix[np.concatenate((c_inds,c_inds+num_sf*2),0),:],0))
plt.xlabel('Pattern ID')
plt.ylabel('Error')
plt.title('Error rates by pattern')
plt.show()

# plt.plot(feat_mat_norm[0][c_inds,20:150],'k.');
# plt.plot(feat_mat_norm[1][c_inds,20:150],'c.')
#plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat)
#plt.plot(label_1nn)
#classifier = KNeighborsClassifier(n_neighbors=1)  
#classifier.fit(vals_feat_subset, label_feat_subset) 



