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

wdir    = r"D:\TB_data\20190406_beer_classification"
fname   = r'chem_ts_feat'

# Optional user inputs:
rec_f = 7642
n_vars = 3 # Variables measured, e.g. if just V,P,T => 3, for V,dV,P,T => 4
# num_feats = 10 # Set a number of features to include for further analysis
# norm = 1 # Set to 1 to normalize all data by feature range to [0,1]
lim_feat= 0 # Sets a limit of 5000 features if set to 1
num_sens = 8 # Goes from 1 to 8, number of sensors on board
# num_reps = 100 # Number of repeat validations for knn
# train_frac = 0.5 # Fraction of traces to use for knn training/testing

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

load_settings()

# Load the data dictionaries for chemicals, features and labels
n_feats,c_dict,f_dict,l_dict = load_TS_feat_mfile(wdir,fname)

# Gather data into matrices and optionally normalize feature array
c_names,n_chems,n_traces,n_msgs,labels,feat_mat = get_arrays(c_dict,f_dict,l_dict)
check_feat_freq(len(feat_mat[0]),rec_f) # error check the user rec_f input

# Generate 1-out 1 knn accuracy by feature, discriminating all chemicals at once
accuracy_vec = chem_acc_vec(n_traces,feat_mat,lim_feat=lim_feat)

# Find most useful features and most frequent feature functions
thresh,c_inds = get_best_feats(accuracy_vec) # or add num_feats=10,20...
most_freq_feats,n_repeats = get_recur_feat_inds(accuracy_vec.argsort()[::-1])

# Turn the feature accuracy into a list of chunked accuracies by V,P,T...etc
acc_chunks = chunk_vector(accuracy_vec,rec_f)

# Concatenate a subset array of useful features and plot MDS
label_feat,concat_arr = concat_sub_feats(feat_mat,c_inds)
arr_trans = MDS_plot(feat_mat,labels,c_inds)

# Get multifeature knn and randomized label knn
print('True labels: ')
acc_tests = rep_knn_mult(concat_arr,label_feat)# n=1,num_reps=100,train_frac=0.5

# Generate true accuracy for every feature and every pattern, n_traces may vary
# accuracy_matrix = mult_acc_matrix(n_traces,labels,feat_mat)

# Perform multifeature knn by chemical
# sing_accuracy_vec = sing_chem_acc_vec(n_traces,feat_mat,lim_feat=lim_feat)
# sing_chem_labels = gen_sing_chem_labels(n_traces)
# binary_chem_acc = sing_chem_knn(sing_chem_labels,feat_mat,num_feats,n_chems)
    
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

# Plot and recover the maximum feature accuracy for every variable for every sensor
max_acc_VPT_HL = {}
all_maxes = np.array([])
for sens in range(1,num_sens + 1): # Goes from 1 to 8, not 0 to 7
    max_acc_VPT_HL[sens] = plot_sensor_chunks(acc_chunks,n_vars,sens)
    all_maxes  = np.append(all_maxes,max_acc_VPT_HL[sens])
plt.show() # indent this to show individual sensor profiles

# Break out feature accuracies by High/Low
HL = 2
for i in range(HL):
    plt.subplot(1,HL,i+1)
    plt.title('H/L comparison')
    plt.boxplot(np.sort(all_maxes[i::HL])[::-1])
    plt.ylim((0,1))
plt.show()

# Break out the feature accuracies by variable
for var in range(n_vars):
    plt.subplot(1,n_vars,var+1)
    plt.title(var)
    plt.stem(all_maxes[var::n_vars])
    plt.ylim((0,1))
    plt.xlim((-1,num_sens*HL))
    # plt.ylabel('Best feature performance')
    plt.xlabel('Sensor HL #')
plt.show()
    
# plot the values for a feature 'f' for all chemicals by label
# f =  np.where(accuracy_vec==np.max(accuracy_vec))[0][0] # the best feature
# plot_feature(f,feat_mat,labels,n_chems)
# np.sort(np.remainder(np.where(accuracy_vec>thresh)[0],9))

# Plot the mean accuracy for only useful features
#plt.stem(1 - np.mean(accuracy_matrix[c_inds,:],0))
#plt.xlabel('Pattern ID')
#plt.ylabel('Error')
#plt.title('Average error rates by pattern, using features above threshold')
#plt.show()
    
# Plot out all the accuracies by feature in linear and log scales:
# line_dist = rec_f
# plot_all_feat_accs(accuracy_vec,line_dist)

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
fs = [11,15]
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
# plt.xlim((-50,50))
                  
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
    



