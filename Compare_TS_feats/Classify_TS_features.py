# -*- coding: utf-8 -*-
"""

Read and classify code for time series trufflebot features
-Eamonn

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import TS_sub_functions # Custom time series sub-functions and processes
from TS_sub_functions import *
 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# User inputs:

wdir    = r"E:\TB_data\3_chem_mult_pattern"
fname   = r'chem_ts_feat_long'
norm    = 0 # Set to 1 to normalize all data by feature range to [0,1]
pl_num  = 0 # increase this to plot more graphs, or 0 for no func plots
thresh  = 0.99 # Set required accuracy for a feature to be considered 'useful'
lim_feat= 1 # Sets a limit of 5000 features if set to 1
train_samp = 30 # set a number of training sample observations for knn 

#concs = {}
#concs['EtOH']    = np.array([0, 347.3,868.2, 1736.5, 2604.7, 3473.0, 4341.2])
#concs['Acetone'] = np.array([0, 1805, 4512,  9024,   13536,  18048,  22561])

#3chem inds: array([    1,     2,     3,     4,     5,     6,     7,     8,     9,
#          15,  3194,  3199,  7640, 22941, 22944, 30567]

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

load_settings()

# Load the data
num_feats,chem_names,feat_dict,label_dict = load_TS_feat_mfile(wdir,fname)

# infer basic data attributes and optionally normalize data
chem_name,num_chems,num_traces,num_msgs,labels,feat_mat_norm = (
        get_mfile_arrays(num_feats,chem_names,feat_dict,label_dict,norm))

# Generate 1-out 1 knn accuracy for every feature by chemical
accuracy_vec = chem_acc_vec(num_traces,feat_mat_norm,pl_num,lim_feat)

c_inds = np.where(accuracy_vec>thresh)[0] # Find useful features (above threshold)

# Concatenate a subset array of useful features
label_feat,concat_arr = concat_sub_feats(feat_mat_norm,c_inds)

MDS_plot(feat_mat_norm,labels,c_inds)

# Generate true accuracy for every feature and every pattern, num_traces may vary
accuracy_matrix = mult_acc_matrix(num_traces,labels,feat_mat_norm,pl_num,lim_feat)

# Get multifeature knn
num_reps = 100 ; n = 1 ; # knn settings
print('True labels: ')
acc_tests = rep_knn_mult(num_reps,train_samp,concat_arr[:,:],label_feat,n)

# develop a null set with random chemical labels
print('Pseudo labels: ')
pseudo_labels = np.random.randint(2,size=len(label_feat))
pseudo_acc_tests = rep_knn_mult(num_reps,train_samp,concat_arr[:,:],pseudo_labels,n)

#%%

# plot the first feature for all chemicals for all patterns
#if pl_num > 0:
#    for c in range(num_chems):        
#        plt.scatter(labels[c],feat_mat_norm[c][0,:])   
#    plt.xlabel('message ID')
#    plt.ylabel('Feature values')
#    plt.show()
   
#pseudo_acc_means = gen_pseudo_mat(
#        num_chems,N,num_msgs,num_traces,labels,feat_mat_norm,lim_feat) 
#
#nf = np.shape(pseudo_acc_means[0])[0]
#num_below_pc = np.zeros(N)
#for k in range(N):
#    plt.loglog(np.array(range(nf)),pseudo_acc_means[k],'c.')
#    num_below_pc[k] = len(np.where(pseudo_acc_means[k]<(1-thresh))[0])
## plt.loglog(np.array(range(nf)),sort_mat_err(accuracy_matrix),'k.')
#plt.xlabel('Features sorted by most accurate first')
#plt.ylabel('Classification error')
#plt.title('True vs. pseudo label feature accuracy comparison - black is true case')
#plt.xlim((10,10000))
#plt.ylim((.001,1))
#plt.grid()
#plt.show()

#for i in range(len(c_inds)):
#    print('Feature ',c_inds[i],':',np.sum(accuracy_matrix[c_inds[i],:]==1),'/',num_msgs,'patterns had all repetitions correctly identified')


# np.sort(np.remainder(np.where(np.mean(accuracy_matrix,1)>0.8)[0],22))

# ----------------------------------------------------------------------------    
#%%
# Output various plots 
# Plot the pseudo and true knn test outputs
plt.hist(pseudo_acc_tests,bins=20,range=(0,1))
plt.hist(acc_tests,bins=20,range=(0,1))
plt.ylabel('Count')
plt.xlabel('Accuracy')
plt.legend
plt.show()

# Plot the mean accuracy for only these useful features
plt.stem(1 - np.mean(accuracy_matrix[c_inds,:],0))
plt.xlabel('Pattern ID')
plt.ylabel('Error')
plt.title('Average error rates by pattern, using features above threshold')
plt.show()

#for feat_ind in range(1):# range(len(c_inds)):
#    
#    chem = {}
#    for c in range(num_chems):
#        chem[c] = feat_mat_norm[c][c_inds[feat_ind],:]    
#        plt.plot(labels[c],chem[c],'.')
#    
#    plt.xlabel('Pattern ID')
#    plt.ylabel('Feature value')
#    plt.title('Plot of the final discriminating feature values by chemical-color')
#    plt.show()
#    

plt.semilogy(1 - accuracy_vec,'.')
plt.xlabel('Feature index, V = 1...N,...')
plt.ylabel('Error rate on chemical identification')
plt.xlim((0,num_feats))
plt.ylim((.001,1))
plt.title('Single feature error identification rates')
plt.show()

# Plot all the feature error rates on log scale
plt.plot(c_inds,1 - accuracy_vec[c_inds],'o')
plt.plot(1 - accuracy_vec,'.')
plt.xlabel('Feature index, V = 1...N,...')
plt.ylabel('Error rate on chemical identification')
plt.xlim((0,num_feats)) # num_feats
plt.ylim((-.01,.1))
plt.title('Single feature error identification rates')
# plt.savefig('Chemical_classification_byfeat.eps', format='eps', dpi=100)
plt.show() 

# Look for recurring features
recurring_feats = np.remainder(c_inds,7642)
obs_reps = np.histogram(recurring_feats,bins=7642,range=(0,7642))
plt.bar(obs_reps[1][1::],obs_reps[0])
plt.xlabel('Recurrent feature index')
plt.ylabel('Feature observation count')
plt.show()


# Plot features with color for label and marker for chemical
fs = [2,3]
lbl_concat = gen_label_concat(labels)
markers = "so^x."
for c in labels.keys():
    inds = np.where(label_feat==c)[0]
    m = markers[c]
    plt.scatter(concat_arr[inds,fs[0]],concat_arr[inds,fs[1]],c=lbl_concat[0:int(num_traces[c])],marker = m)
plt.legend(chem_names)
plt.xlabel('Feature #' + str(fs[0]))
plt.ylabel('Feature #' + str(fs[1]))
#threshold = np.array([.9,.95,.98,.99,.995])
#threshold = np.linspace(.7,1,1000)
#acc_feats = np.zeros(len(threshold))
#tick = 0
#for j in threshold:
#    acc_feats[tick] = np.sum(np.mean(accuracy_matrix,1)>=j)
#    tick = tick+1

#plt.loglog(1-threshold,acc_feats,'.')
#plt.ylabel('# Features classifying at < error')
#plt.xlabel('Identification error rate')
#plt.ylim((1,1000))
#plt.xlim((.001,1))
#plt.grid()
#plt.show() 

# plt.plot(feat_mat_norm[0][c_inds,20:150],'k.');
# plt.plot(feat_mat_norm[1][c_inds,20:150],'c.')
#plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat)
#plt.plot(label_1nn)
#axes = 3
#il = np.zeros(axes)
#for i in range(axes):
#    il[i] = random.choice(range(len(c_inds)))
#from mpl_toolkits.mplot3d import axes3d
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, axisbg="0.5")
#ax = fig.gca(projection='3d')
#ax.scatter(concat_arr[:,il[int(0)]],concat_arr[:,int(il[1])],concat_arr[:,int(il[2])],c=label_feat,s=100,cmap='Spectral')
#plt.show()
#for i in range(4):
#    f1 = random.choice(range(len(c_inds)))
#    f2 = random.choice(range(len(c_inds)))
#    plt.scatter(concat_arr[:,f1],concat_arr[:,f2],c=label_feat,cmap='Spectral',vmin=-4,vmax=3)
#    plt.xlabel(c_inds[f1])
#    plt.ylabel(c_inds[f2])
#    plt.show()
    



