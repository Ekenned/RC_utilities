# -*- coding: utf-8 -*-
"""

Basic read in code for time series features
-Eamonn

"""

# Standard scientific package imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Time series sub-functions and processes
import TS_sub_functions
from TS_sub_functions import *

load_settings()
#------------------------------------------------------------------------------
#%%

wdir = r"C:\Users\Eamonn\Documents\GitHub\RC_utilities\Compare_TS_feats\example_feat_data"
fname = r'chem_ts_feat'
num_vars = 6 # number of test variables used
normalize = 1

# Load the data and the basic data sizes, attributes
# Also perform a normalization by feature, across all chemicals (blindly)
chem_name,num_chems,num_feats,num_traces,num_msgs,labels,feat_mat_norm = (
        load_TS_feat_matfile(wdir,fname,normalize))

#-----------------------------------------------------------------------------
#%%

acc = np.zeros(num_msgs)
accuracy_matrix = np.zeros((num_feats,num_msgs))
plot_feat_num = 1 # increase this to plot more graphs
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
            x,y = get_best_subplot(num_msgs)
            plt.subplot(y,x,trace_num+1)
            plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat,vmin=-.5,vmax=1.5)
            plt.title(trace_num) 
            # plt.ylim((0,1))
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

# ----------------------------------------------------------------------------    
#%%
# Output various plots 
    
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



