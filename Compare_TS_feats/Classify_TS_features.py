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
# Required user inputs:

wdir    = r"E:\TB_data\20190321_3chem_low_conc\features"
fname   = r'chem_ts_feat_low_conc_9f'

# Optional user inputs:

num_feats = 10 # Set a number of features to include for further analysis
# norm = 1 # Set to 1 to normalize all data by feature range to [0,1]
# lim_feat= 1 # Sets a limit of 5000 features if set to 1
# pl_num = 1 # increase this to plot more graphs, or 0 for no func plots
# num_reps = 100 # Number of validations for knn
# train_frac = 0.5 # Fraction of traces to use for knn training/testing

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

load_settings()

# Load the data dictionaries for chemicals, features and labels
n_feats,c_dict,f_dict,l_dict = load_TS_feat_mfile(wdir,fname)

# Gather data into matrices and optionally normalize feature array
c_names,n_chems,n_traces,n_msgs,labels,feat_mat = get_arrays(c_dict,f_dict,l_dict)

# Generate 1-out 1 knn accuracy for every feature discriminating N chemicals
accuracy_vec = chem_acc_vec(n_traces,feat_mat)

thresh,c_inds = get_best_feats(accuracy_vec,num_feats) # Find useful features (above threshold)

# Concatenate a subset array of useful features
label_feat,concat_arr = concat_sub_feats(feat_mat,c_inds)

# Concatenate a subset array of useful features
MDS_plot(feat_mat,labels,c_inds)

# Generate true accuracy for every feature and every pattern, n_traces may vary
accuracy_matrix = mult_acc_matrix(n_traces,labels,feat_mat)

# Get multifeature knn and randomized label knn
print('True labels: ')
acc_tests = rep_knn_mult(concat_arr,label_feat)# n=1,num_reps=100,train_frac=0.5

# develop a null set with random chemical labels
print('Pseudo labels: ')
pseudo_labels = np.random.randint(2,size=len(label_feat))
pseudo_acc_tests = rep_knn_mult(concat_arr,pseudo_labels)


#%% Output various plots 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Plot the accuracies for num_rep training splits for true and pseudo label knn:
hist_acc(pseudo_acc_tests)
hist_acc(acc_tests)
plt.show()

# plot the values for a feature 'f' for all chemicals by label
f =  np.where(accuracy_vec==np.max(accuracy_vec))[0][0] # the best feature
plot_feature(f,feat_mat,labels,n_chems)
# np.sort(np.remainder(np.where(accuracy_vec>thresh)[0],9))

# Plot the mean accuracy for only useful features
plt.stem(1 - np.mean(accuracy_matrix[c_inds,:],0))
plt.xlabel('Pattern ID')
plt.ylabel('Error')
plt.title('Average error rates by pattern, using features above threshold')
plt.show()
    
# Plot out all the accuracies by feature in linear and log scales:
line_dist = 72
plot_all_feat_accs(accuracy_vec,line_dist)

# Look for recurring features
rec_f = 9 # number of features measured per variable
recurring_feats = np.remainder(c_inds,rec_f)
obs_reps = np.histogram(recurring_feats,bins=rec_f,range=(0,rec_f))
plt.bar(obs_reps[1][1::],obs_reps[0])
plt.xlabel('Recurrent feature index')
plt.ylabel('Feature observation count')
plt.show()

#repnum = 36 # Just a feature group size repeat one the figures over
#max_val = np.zeros(repnum)
#for i in range(repnum):
#    max_val[i] = np.max(accuracy_vec[i::repnum])
#plt.stem(max_val)
##for i in range(4):
##    plt.plot([i*9-.5,i*9-.5],[0, 1],'k')
#plt.ylim((0,1))
#plt.xlabel('Feature repeat #')
#plt.ylabel('Max accuracy')
#plt.show()
   
# Plot 2 features with color for label and marker for chemical
fs = [0,1]
lbl_concat = gen_label_concat(labels)
markers = "so^x."
for c in labels.keys():
    inds = np.where(label_feat==c)[0]
    m = markers[c]
    # plt.scatter(concat_arr[inds,fs[0]],concat_arr[inds,fs[1]],c=lbl_concat[0:int(n_traces[c])],marker = m)
    plt.scatter(concat_arr[inds,fs[0]],concat_arr[inds,fs[1]],marker = m)
plt.legend(c_names)
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

# plt.plot(feat_mat[0][c_inds,20:150],'k.');
# plt.plot(feat_mat[1][c_inds,20:150],'c.')
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
    



