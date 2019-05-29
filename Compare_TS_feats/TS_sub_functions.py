# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:30:09 2019

@author: Eamonn

"""
import os
import numpy as np
import matplotlib as mpl
import scipy.io as sio
import matplotlib.pyplot as plt
import platform
from functools import reduce
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier
import random
import time

#import h5py
#import pandas as pd
#from sklearn import tree
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score
#from sklearn import linear_model
#from scipy.ndimage.filters import maximum_filter1d,median_filter
#from scipy.ndimage.measurements import label
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestRegressor

# remove nan and inf entries in an array
def zero_nan_and_infs(array):
    array[np.isnan(array)] = 0 
    array[~np.isfinite(array)] = 0 # remove infinite entries
    return array

# Return the indices of the highest values in an array
def get_high_value_inds(vector,num_inds):
    if num_inds>len(vector): # default setting of return all sorted inds
        num_inds = len(vector)
    return vector.argsort()[::-1][0:num_inds]

# Error check that a vector is an integer multiple of a number of features
def check_feat_freq(vec_length,rec_f):
    if np.round(vec_length/rec_f) != vec_length/rec_f:
        print('rec_f number is wrong or feat mat is wrong size...')
        time.sleep(1)
        return

# Load time series dictionaries from mat file
def load_TS_feat_mfile(wdir,fname):
    
    # Load the data
    os.chdir(wdir)
    TSF_data = sio.loadmat(fname + '.mat')
    label_dict = TSF_data[fname]['labels'][0][0]
    feat_dict = TSF_data[fname]['feat_mat'][0,0]
    chem_names = TSF_data[fname]['chem_names'][0,0]
    # num_chems = np.shape(chem_names)[0]
    num_feats = np.shape(feat_dict[0][chem_names[0][0][0]][0])[0]
    
    return num_feats,chem_names,feat_dict,label_dict

# Get basic properties and feature array out of dictionaries
def get_arrays(chem_names,feat_dict,label_dict,norm=0):
    
    num_feats = np.shape(feat_dict[0][0][0])[0] # get the # features per trace
    
    # Core information, pattern labels, features, chemicals for dictionaries
    labels = {}
    feat_mat = {}
    chem_name = {}
    
    # And some ancillary information for normalization
    num_chems = np.shape(chem_names)[0]
    num_traces = np.zeros(num_chems)
    feat_max_arr = np.zeros((num_feats,num_chems))
    feat_min_arr = np.zeros((num_feats,num_chems))
    
    # Get the labels and feature matrix for every chemical
    for c in range(num_chems):
    
        chem_name[c] = chem_names[c][0][0]
        feat_mat[c] = feat_dict[chem_name[c]][0,0]
        feat_mat[c]= zero_nan_and_infs(feat_mat[c])  # remove nan / inf entries
        labels[c] = label_dict[chem_name[c]][0][0]
        num_traces[c] = np.shape(feat_mat[c])[1]
        feat_max_arr[:,c] = np.max(feat_mat[c],1)
        feat_min_arr[:,c] = np.min(feat_mat[c],1)
        num_msgs = len(np.unique(labels[0]))    
    
    # Normalize data from 0 - 1 
    abs_max = np.nanmax(feat_max_arr,1) # max and min over all chemicals by feature
    abs_min = np.nanmin(feat_min_arr,1)
    range_feat = abs_max - abs_min + 10**-6
    feat_mat_norm = feat_mat # instantiate the normalized vector
    
    if norm == 1:
        
        for c in range(num_chems):
            
            for f in range(num_feats):
                all_trace_vals = feat_mat[c][f,:]
                all_trace_vals = (all_trace_vals - abs_min[f]) / range_feat[f]
                feat_mat_norm[c][f,:] = all_trace_vals 
    
    return chem_name,num_chems,num_traces,num_msgs,labels,feat_mat_norm

# Iterate 1D 1NN over all points and test against the true labels
def knn1d(num_obs,vals,labels):
    label_1nn = np.zeros(num_obs)
    for i in range(num_obs):
        distances = np.abs(vals - vals[i]) # distance vector from ith value
        distances[i] = np.max(distances) + 1 # rule out the value itself
        # determine the label associated with the minimum distance value
        label_1nn[i] = labels[np.where(distances==np.min(distances))[0][0]]
    # Find the fraction of instances where this label equals the true label
    sing_acc = np.sum(1*(labels==label_1nn))/num_obs
    return sing_acc    

# Plot sorted accuracy matrix mean errors
def sort_mat_err(mat):
    mean_errs = 1 - np.sort(np.mean(mat,1))[::-1]
    return mean_errs

def sort_vec_err(vec):
    errs = 1 - np.sort(vec)[::-1]
    best_inds = accuracy_vec.argsort()[::-1]
    return errs,best_inds

# Chunk a vector into a list of blocks of length chunk_size
def chunk_vector(vec,chunk_size):
    check_feat_freq(len(vec),chunk_size) # check the chunk size divides to int
    mat_dim = (np.int(len(vec)/chunk_size),chunk_size)# get output size
    vec_blocks = np.reshape(vec,mat_dim) # create the output matrix
    return vec_blocks

# Concatenate all chemical data, reduce matrix to features 'c_inds' of interest
def concat_sub_feats(feat_mat_norm,c_inds):
    
    num_chems = len(feat_mat_norm.keys())
    feat_arr = {}
    concat_arr = np.array([])
    label_feat = np.array([])
    
    for c in range(num_chems):
        feat_arr[c] = feat_mat_norm[c][c_inds,:] # c_inds in first
        label_list = c*(np.ones(np.shape(feat_arr[c])[1]))
        label_feat = np.append(label_feat,label_list)
        if c == 0:
            concat_arr = feat_arr[c]
        else:
            concat_arr = np.concatenate((concat_arr,feat_arr[c]),1)       
    concat_arr = np.transpose(concat_arr)
            
    return label_feat,concat_arr

# Look for recurring features, choose best 'num_check' subset for consideration
def get_recur_feat_inds(inds,rec_f=7642,num_check=500):
    
    inds = inds[0:num_check]
    plotting = 1
    # Find recurring indices within the feature set
    recurring_feats = np.remainder(inds,rec_f)
    obs_reps = np.histogram(recurring_feats,bins=rec_f,range=(0,rec_f))
    
    # Refine to a list of only those features which repeat more than once
    feat_repeat_inds = np.where(obs_reps[0]>1)[0]
    repeat_count = obs_reps[0][feat_repeat_inds]
    feats_sorted_by_freq = feat_repeat_inds[repeat_count.argsort()[::-1]]
    sorted_repeat_count = repeat_count[repeat_count.argsort()[::-1]]
    
    if plotting == 1:
        plt.stem(feats_sorted_by_freq,sorted_repeat_count)
        plt.xlabel('Recurrent feature index')
        plt.ylabel('Feature observation count')
        plt.show()
    
    return feats_sorted_by_freq,sorted_repeat_count

# Get reduced multidimensional representation of the data
def MDS_plot(feat_mat_norm,labels,c_inds):
    
    label_feat,concat_arr = concat_sub_feats(feat_mat_norm,c_inds)
    
    embedding = MDS(n_components=2)
    
    arr_trans = embedding.fit_transform(concat_arr)
    
    marker_list =	{0: "^",1: "o",2: "s",3: "o",4: "x"}
    for c in range(len(feat_mat_norm.keys())):
        inds = np.where(label_feat==c)[0]
        plt.scatter(arr_trans[inds,0],arr_trans[inds,1],marker=marker_list[c])
    plt.title('2-component reduction of discriminating features')
    plt.show()
    return arr_trans
    
# Generate a concatenated list of N-chemical ID labels for traces [0,0,0...N,N,N]
def gen_chem_labels(num_traces):
    chem_labels = np.array([])
    for c in range(len(num_traces)):
        chem_labels = np.append(chem_labels,c*(np.ones(int(num_traces[c]))))
    return chem_labels

# Generate a dictionary of binary presence/absence for every chemical
def gen_sing_chem_labels(num_traces):
    chem_labels = gen_chem_labels(num_traces)
    sing_chem_labels = {}
    for c in range(len(num_traces)):  
        sing_chem_labels[c] = 1*(chem_labels == c)  
    return sing_chem_labels

# Gather and append all trace labels for all chemicals into a 1D vector
def gen_label_concat(labels):
    lbl_concat = np.array([])
    for c in labels.keys():
        lbl_concat = np.append(lbl_concat,labels[c])
    return [int(i) for i in lbl_concat]

# Gather and append all values for 1 feature 'f' for all chemicals
def append_chem_vals(feat_mat_norm,f):
    chem_vals= np.array([])
    for c in range(len(feat_mat_norm.keys())):
        chem_vals = np.append(chem_vals,feat_mat_norm[c][f,:])
    return chem_vals

# Generate an accuracy for N chemicals by feature
def chem_acc_vec(num_traces,feat_mat_norm,plot_feat_num=0,lim_feat=0,feat_lim=5000):
    
    chem_labels = gen_chem_labels(num_traces) # chem labels
    randomize_labels = 0
    if randomize_labels == 1: # optionally randomize labels for negative control
        np.random.shuffle(chem_labels)
        
    # num_chems = len(num_traces) # len(feat_mat_norm.keys())
    num_feats = np.shape(feat_mat_norm[0])[0] # just define this internally
    net_obs = int(sum(num_traces)) # total trace observations for all chems
    
    if lim_feat == 1: # optionally limit to 5000 features for time saving, etc
        if num_feats>feat_lim:
            num_feats=feat_lim

    accuracy_vec = np.zeros(num_feats)
    
    for f in range(num_feats):
        
        # Append all features values for all chemicals, perform knn calculation    
        vals_feat = append_chem_vals(feat_mat_norm,f)
        accuracy_vec[f] = knn1d(net_obs,vals_feat,chem_labels)

        # Plotting and printing
        if plot_feat_num>0: 
            # if its equal to the best feature yet, and above 70% accurate, plot it
            if accuracy_vec[f]>=np.max([np.max(accuracy_vec),0.7]):
                plt.scatter(range(len(vals_feat)),vals_feat,c=chem_labels,vmin=-.5,vmax=1.5)
                plt.title(f) 
                plt.xlabel('Trace #, all chems')
                plt.ylabel('Feature value')
            plt.show()

        if np.remainder(f,100)==0: 
            print(f,'/',num_feats,', chem acc: ', accuracy_vec[f])
            
    # Data cleaning
    accuracy_vec = zero_nan_and_infs(accuracy_vec) # remove nans/infs            
    return accuracy_vec

# Generate an accuracy for individual chemicals by feature
def sing_chem_acc_vec(num_traces,feat_mat_norm,lim_feat=0,feat_lim=5000):
    
    chem_labels = gen_sing_chem_labels(num_traces) # dict of single chem labels
        
    num_chems = len(num_traces) # len(feat_mat_norm.keys())
    num_feats = np.shape(feat_mat_norm[0])[0] # just define this internally
    net_obs = int(sum(num_traces)) # total trace observations for all chems
    
    if lim_feat == 1: # optionally limit to 5000 features for time saving, etc
        if num_feats>feat_lim:
            num_feats=feat_lim

    accuracy_vec= np.zeros((num_feats,num_chems))
    inds_discr  = np.zeros((num_feats,num_chems))
    
    for f in range(num_feats):
        
        # Append all features values for all chemicals   
        vals_feat = append_chem_vals(feat_mat_norm,f)
        
        # perform knn calculation broken out for every chemical label-set
        for c in range(num_chems):
            accuracy_vec[f,c] = knn1d(net_obs,vals_feat,chem_labels[c])

        if np.remainder(f,100)==0: 
            print(f,'/',num_feats,', max feat acc: ', np.max(accuracy_vec[f,0:num_chems]))
            
    
    for c in range(num_chems):
        inds_discr[:,c] = get_high_value_inds(accuracy_vec[:,c],num_feats)
    # Data cleaning
    accuracy_vec = zero_nan_and_infs(accuracy_vec) # remove nans/infs            
    return accuracy_vec
    
def get_best_feats(accuracy_vec,n_feats=10):
    if len(accuracy_vec)<n_feats: # default to all features if < n_feats
        n_feats = len(accuracy_vec)
    thresh = np.sort(accuracy_vec)[::-1][n_feats]
    return thresh,np.where(accuracy_vec>=thresh)[0] # Find useful features (above threshold)

# Run knn tests for binary single chemical accuracies
def sing_chem_knn(sing_chem_labels,feat_mat,num_feats,n_chems):

    # This will fill with test/train knn accuracy for every feature, every chem
    binary_chem_acc = np.zeros((num_feats,n_chems))
    for f in range(1):
        print(f)
        f = f + 1
        sing_c_inds = np.zeros((f,n_chems))
        for c in range(n_chems):
            
            # Generate best indices, labels, and subset array for each chemical seperately
            sing_c_inds[:,c] = inds_discr[0:f,c].astype(int)
            sing_labels = sing_chem_labels[c]
            alt_lbl,concat_arr = concat_sub_feats(feat_mat,sing_c_inds[:,c].astype(int))
            
            # Run the knn classifier
            print('Chemical #',c,':')
            acc_tests = rep_knn_mult(concat_arr,sing_labels)
            binary_chem_acc[f-1,c] = 1 - np.median(acc_tests)
            
    return binary_chem_acc

# Repeat knn with different randomized training labels for validation   
def knn_mult(train_frac,concat_arr,label_feat,n):
    
    # train_samp = 500; n = 3
    classifier = KNeighborsClassifier(n_neighbors = n)  
    ns = np.shape(concat_arr)
    train_samp = np.round(train_frac*ns[0]).astype(int)
    train_lbls = np.random.choice(ns[0], size = train_samp, replace=False)
    test_lbls = np.setdiff1d(range(ns[0]),train_lbls)
    
    classifier.fit(concat_arr[train_lbls,:], label_feat[train_lbls]) 
    y = classifier.predict(concat_arr[test_lbls,:])
    # y_net = classifier.predict(concat_arr[test_lbls,:])
    acc_test = np.sum(y == label_feat[test_lbls]) / (ns[0] - train_samp)
    # acc_net = np.sum(y_net == label_feat) / ns[0]
    return acc_test # ,acc_net

# Repeat multifeature knn N times
def rep_knn_mult(concat_arr,label_feat,n=1,num_reps=100,train_frac=0.5):
    acc_tests = np.zeros(num_reps)
    for i in range(num_reps):
        acc_tests[i] = knn_mult(train_frac,concat_arr,label_feat,n)
    print('Classification error: ',1 - np.median(acc_tests))
    # print(' \n ')
    return acc_tests

# split indices randomly into test and train groups
def test_train_inds(n_obs,train_split):
    
    n_train       = round(train_split*n_obs)
    n_test        = n_obs - n_train   
    train_inds   = (np.random.choice(n_obs, n_train, replace=False)).astype(int)
    test_inds    = (np.setdiff1d(range(n_obs),train_inds)).astype(int)

    return train_inds,test_inds
   
# Generate entire accuracy matrix for some pseudo labels, repeat N times, return means
#def gen_pseudo_mat(num_chems,num_msgs,num_traces,N,labels,feat_mat_norm,lim_feat=0):
#
#    pseudo_acc_means = {}
#    for k in range(N):
#        print('Developing pseudo-accuracies for label set #',k,'/',N-1)
#        
#        # Perform a pseudo random label bootstrap, first develop the random labels
#        pseudo_labels = {}
#        for i in range(num_chems):
#            pseudo_labels[i] = 1 + np.random.choice(int(num_msgs),int(num_traces[i]))
#        
#        # Applt the pseudo labels to develop an accuracy matrix, and mean vector
#        pseudo_acc_mat = mult_acc_matrix(num_traces,pseudo_labels,feat_mat_norm,0,lim_feat)
#        pseudo_acc_means[k] = sort_mat_err(pseudo_acc_mat)
        
    return pseudo_acc_means

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))) 
    
def get_best_subplot(num_plots):
    facs = np.array(list(factors(num_plots)))
    minima = num_plots
    for i in facs:
        for j in facs:
            if i*j==num_plots:
                if abs(i-j) < minima:
                    minima = abs(i-j)
                    x_b = i
                    y_b = j
                    
    return x_b,y_b
        
def load_settings():
    
    # Style options for figures
    plt.style.use(['seaborn-paper']) # ,'dark_background'])
    a = platform.uname()
    # print(a)
    if a[1] == 'XPS15': # settings for graphs on 4K XPS15 screen only
        print('Using 4k figure settings...')
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['figure.figsize']   = (24, 20)
        mpl.rcParams['axes.titlesize']   = 30
        mpl.rcParams['axes.labelsize']   = 30
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 20
        mpl.rcParams['xtick.labelsize']  = 30
        mpl.rcParams['ytick.labelsize']  = 30
        mpl.rcParams['axes.linewidth'] = 3
    else:
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['figure.figsize']   = (10, 10)
        mpl.rcParams['axes.titlesize']   = 20
        mpl.rcParams['axes.labelsize']   = 20
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['xtick.labelsize']  = 16
        mpl.rcParams['ytick.labelsize']  = 16
        mpl.rcParams['axes.linewidth'] = 3
        
    print('Modules and settings loaded')
    
# Need to generalize this to N chems, with median sorted 
def check_2chem_sep(vals,labels):
    c1_vals = vals[labels==0]
    c2_vals = vals[labels==1]
    sep = ((max(c2_vals)<min(c1_vals)) or (max(c1_vals)<min(c2_vals)))
    return sep

# Generate an accuracy for every feature and pattern for N chemicals
def mult_acc_matrix(num_traces,labels,feat_mat_norm,plot_feat_num=0,lim_feat=0,feat_lim=5000):

    num_feats = np.shape(feat_mat_norm[0])[0] # just define these internally
    if lim_feat == 1: # optionally limit to 5000 features for time, etc
        if num_feats>feat_lim:
            num_feats=feat_lim
    num_msgs = len(np.unique(labels[0]))
    acc = np.zeros(num_msgs)
    check_sep = acc
    accuracy_matrix = np.zeros((num_feats,num_msgs))
    num_chems = len(feat_mat_norm.keys())
    
    for f in range(num_feats):

        if f == round(num_feats/2):
            print('Halfway through features... (', f,'/',num_feats,')')
        
        for trace_num in range(num_msgs):
            
            # Seperate out features by chemical
            v = {}
            num_obs = np.zeros(num_chems)
            vals_feat = np.array([])
            label_feat = np.array([]) # chemical labels
            for c in range(num_chems):
                # for the fth feature, take only those traces where the label is = value
                v[c] = feat_mat_norm[c][f,np.where(labels[c]==trace_num+1)[0]]
                num_obs[c] = len(v[c])
                vals_feat = np.append(vals_feat,v[c])
                label_feat = np.append(label_feat,c*(np.ones(int(num_obs[c]))))
            net_obs = int(sum(num_obs))

            if f<plot_feat_num: 
                x,y = get_best_subplot(num_msgs)
                plt.subplot(y,x,trace_num+1)
                # plt.scatter(label_feat,vals_feat,c=label_feat)
                plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat,vmin=-.5,vmax=1.5)
                plt.title(trace_num) 
                # plt.ylim((0,1))
                plt.yticks([])
            
            acc[trace_num] = knn1d(net_obs,vals_feat,label_feat)
            
            #if num_chems==2:
            #    check_sep[trace_num] = check_2chem_sep(vals_feat,label_feat)  

        if np.remainder(f,10)==0: 
            if np.mean(acc)>0:
                print(f,', acc: ', np.mean(acc))
        plt.show()
        
        accuracy_matrix[f,:] = acc
    # print('Feature x Pattern Accuracy matrix complete')
    accuracy_matrix = zero_nan_and_infs(accuracy_matrix)  
    return accuracy_matrix

#################################################################################
# Plotting functions

def plot_sensor_chunks(acc_chunks,n_vars,sens,f_plt=500,HL=2):
    best_acc_chunk = np.zeros((n_vars,HL))
    for L in range(HL):
        plt.subplot(1,2,L+1)
        if L == 0:
            plt.title('High voltage')
        else:
            plt.title('Low voltage')
        best_acc_chunk[:,L] = plot_sing_chunk(acc_chunks,n_vars,sens,L,f_plt,HL)
    # plt.show() # comment this in to plot single sensors
    return np.transpose(best_acc_chunk)

    # Array like [V, P, T ; V, P, T], if P was bad then:
    # array([[ 0.825     ,  0.54166667,  0.825     ],
    #        [ 0.86666667,  0.58333333,  0.86666667]])

def plot_var_acc(acc_chunks,n_vars=3,num_sens=8,var=0,HL=2,num_mean_feat = 5,plotting=1,rec_f = 219): 

    acc_mean = np.zeros((num_sens,HL))
    chunked_acc = np.array([])
    for sens in range(num_sens): # for every sensor
        for L in range(HL): # for high and low
            select_indices = var + n_vars*L + HL*n_vars*(sens)
            loc_chunk = acc_chunks[select_indices,:]
            chunked_acc = np.concatenate((chunked_acc,loc_chunk),0)
            acc_mean[sens,L] = np.mean(np.sort(loc_chunk)[::-1][0:num_mean_feat])
    if plotting == 1:          
        plt.plot(chunked_acc,'k.')
        for i in range(num_sens*HL):
            plt.plot([rec_f *i,rec_f *i],[0,1],'c--')
            plt.xlabel('Features for variable #' + str(var))
            plt.ylabel('Classification accuracy')
    
    return acc_mean
    
def plot_sing_chunk(acc_chunks,n_vars,sens,L,f_plt,HL):
    colorlist = 'kgcrbm'
    best_acc_chunk = np.zeros(n_vars)
    for i in range(n_vars):
        chunked_feats = np.sort(
                acc_chunks[i + n_vars*L + HL*n_vars*(sens-1),:])[::-1][0:f_plt]
        best_acc_chunk[i] = np.mean(chunked_feats[0:5]) # hacky but works well
        plt.plot(chunked_feats,colorlist[i])
    plt.xlabel('Best feature #')
    plt.ylabel('Classification accuracy')
    plt.ylim((0.2,1))
    plt.xlim((0,f_plt)) 
    return best_acc_chunk
    
def plot_feature(f,feat_mat,labels,n_chems):
    for c in range(n_chems):        
        plt.scatter(labels[c],feat_mat[c][f,:])   
        plt.xlabel('Class label')
        plt.ylabel('Feature value')
        plt.title(f)
    plt.show()

# Plot the pseudo and true knn test outputs
def hist_acc(acc,lin_dist = 1000 ):
    plt.hist(acc,bins=20,range=(0,1))
    plt.ylabel('Validation count')
    plt.xlabel('Accuracy')

# Plot out all the feature accuracies in linear and log scales
def plot_all_feat_accs(accuracy_vec,line_dist):
    
    # Plot all the feature error rates on linear scale
    plt.semilogy(1 - accuracy_vec,'.')
    feat_plot_details(accuracy_vec,line_dist,ylims=[.001,1])
    plt.show()
    
    # Plot all the feature error rates on log scale
    plt.plot(1 - accuracy_vec,'o')
    feat_plot_details(accuracy_vec,line_dist,ylims=[-.01,.3])
    plt.show()
        
def feat_plot_details(accuracy_vec,line_dist,ylims):
    
    plt.xlabel('Feature index, V = 1...N,...')
    plt.ylabel('Error rate on chemical identification')
    plt.xlim((0,len(accuracy_vec)+1)) # num_feats
    plt.ylim(ylims)
    plt.title('Single feature error identification rates')
    i = 0
    while i*line_dist<len(accuracy_vec):
        i += 1
        plt.plot([i*line_dist,i*line_dist],ylims,'k--')

def plot_rec_feats(c_inds,rec_f = 9):      
    # Look for recurring features
    recurring_feats = np.remainder(c_inds,rec_f)
    obs_reps = np.histogram(recurring_feats,bins=rec_f,range=(0,rec_f))
    plt.bar(obs_reps[1][1::],obs_reps[0])
    plt.xlabel('Recurrent feature index')
    plt.ylabel('Feature observation count')
    plt.show()


#################################################################################

def gen_feature_list(good_feats,good_sensors,feats_per_sensor):
    z = np.zeros(good_sensors.size * good_feats.size)
    counter = 0
    for i in feats_per_sensor*good_sensors:
        for j in good_feats:
            z[counter] = np.int(i + j)
            counter += 1
    return z
  
#good_feats = np.array([3,5,30,32,33])
#good_sensors = np.array([0,2,4,6,8,10,12,14,15])
#feats_per_sensor = 36      
#est_feats = gen_feature_list(good_feats,good_sensors,feats_per_sensor).astype(int)

    
# Generate entire accuracy matrix for some pseudo labels N times, return means
#def twochem_pseudo_mat(num_chems,N,num_msgs,num_traces,labels,feat_mat_norm):
#
#    pseudo_acc_means = {}
#    for k in range(N):
#        print('Developing pseudo-accuracies for label set #',k,'/',N-1)
#        
#        # Perform a pseudo random label bootstrap, first develop the random labels
#        pseudo_labels = {}
#        for i in range(num_chems):
#            pseudo_labels[i] = 1 + np.random.choice(int(num_msgs),int(num_traces[i]))
#        
#        # Applt the pseudo labels to develop an accuracy matrix, and mean vector
#        pseudo_acc_mat = two_chem_acc_matrix(num_traces,pseudo_labels,feat_mat_norm,0)
#        pseudo_acc_means[k] = sort_mat_err(pseudo_acc_mat)
#        
#    return pseudo_acc_means
    
# Generate an accuracy for every feature and pattern for only 2 chemicals
#def two_chem_acc_matrix(num_traces,labels,feat_mat_norm,plot_feat_num):
#
#    num_feats = np.shape(feat_mat_norm[0])[0] # just define these internally
#    num_msgs = len(np.unique(labels[0]))
#    acc = np.zeros(num_msgs)
#    accuracy_matrix = np.zeros((num_feats,num_msgs))
#    
#    for f in range(num_feats):
#        
#        if f == round(num_feats/2):
#            print('Halfway through features... (', f,'/',num_feats,')')
#        
#        for trace_num in range(num_msgs):
#            
#            # Seperate out features by chemical, plot a mxn grid for mxn traces
#            v1 = feat_mat_norm[0][f,np.where(labels[0]==trace_num+1)[0]]
#            v2 = feat_mat_norm[1][f,np.where(labels[1]==trace_num+1)[0]]
#            num_obs = len(v1)+len(v2)
#            vals_feat = (np.concatenate((v1,v2),0))
#            label_feat = np.concatenate((np.zeros(len(v1)),np.ones(len(v2))),0)
#            if f<plot_feat_num: 
#                x,y = get_best_subplot(num_msgs)
#                plt.subplot(y,x,trace_num+1)
#                plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat,vmin=-.5,vmax=1.5)
#                plt.title(trace_num) 
#                # plt.ylim((0,1))
#                plt.yticks([])
#            
#            acc[trace_num] = knn1d(num_obs,vals_feat,label_feat)
#           
#        plt.show()
#            
#        accuracy_matrix[f,:] = acc
#    print('Feature x Pattern Accuracy matrix complete')
#    return accuracy_matrix

        
        