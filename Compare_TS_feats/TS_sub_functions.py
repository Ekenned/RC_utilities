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
#from sklearn.manifold import MDS
#from sklearn.neighbors import KNeighborsClassifier  

def load_TS_feat_mfile(wdir,fname):
    
    # Load the data
    os.chdir(wdir)
    TSF_data = sio.loadmat(fname + '.mat')
    label_dict = TSF_data[fname]['labels'][0][0]
    feat_dict = TSF_data[fname]['feat_mat'][0,0]
    chem_names = TSF_data[fname]['chem_names'][0,0]
    num_chems = np.shape(chem_names)[0]
    num_feats = np.shape(feat_dict[0][chem_names[0][0][0]][0])[0]
    
    return num_feats,chem_names,feat_dict,label_dict

def get_mfile_arrays(num_feats,chem_names,feat_dict,label_dict,norm):
    
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
    for chem_num in range(num_chems):
    
        chem_name[chem_num] = chem_names[chem_num][0][0]
        feat_mat[chem_num] = feat_dict[chem_name[chem_num]][0,0]
        feat_mat[chem_num][np.isnan(feat_mat[chem_num])] = 0 # remove nan entries
        labels[chem_num] = label_dict[chem_name[chem_num]][0][0]
        num_traces[chem_num] = np.shape(feat_mat[chem_num])[1]
        feat_max_arr[:,chem_num] = np.max(feat_mat[chem_num],1)
        feat_min_arr[:,chem_num] = np.min(feat_mat[chem_num],1)
        num_msgs = len(np.unique(labels[0]))    
    
    # Normalize data from 0 - 1 
    abs_max = np.nanmax(feat_max_arr,1) # max and min over all chemicals by feature
    abs_min = np.nanmin(feat_min_arr,1)
    range_feat = abs_max - abs_min + 10**-6
    feat_mat_norm = feat_mat # instantiate the normalized vector
    
    if norm == 1:
        
        for chem_num in range(num_chems):
            
            for f in range(num_feats):
                all_trace_vals = feat_mat[chem_num][f,:]
                all_trace_vals = (all_trace_vals - abs_min[f]) / range_feat[f]
                feat_mat_norm[chem_num][f,:] = all_trace_vals 
    
    return chem_name,num_chems,num_traces,num_msgs,labels,feat_mat_norm

# Iterate 1D 1NN over all points and test against the true labels
def knn1d(num_obs,vals,labels):
    label_1nn = np.zeros(num_obs)
    for i in range(num_obs):
        distances = np.abs(vals - vals[i])
        distances[i] = np.max(distances) + 1 # rule out the value itself
        label_1nn[i] = labels[np.where(distances==np.min(distances))[0][0]]
        sing_acc = np.sum(1*(labels==label_1nn))/num_obs
    return sing_acc         

# Generate an accuracy for every feature and pattern for 2 Chemicals
def gen_acc_matrix(num_traces,labels,feat_mat_norm,plot_feat_num):

    num_feats = np.shape(feat_mat_norm[0])[0] # just define these internally
    num_msgs = len(np.unique(labels[0]))
    acc = np.zeros(num_msgs)
    accuracy_matrix = np.zeros((num_feats,num_msgs))
    
    for f in range(num_feats):
        
        if f == round(num_feats/2):
            print('Halfway through features... (', f,'/',num_feats,')')
        
        for trace_num in range(num_msgs):
            
            # Seperate out features by chemical, plot a mxn grid for mxn traces
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
            
            acc[trace_num] = knn1d(num_obs,vals_feat,label_feat)
           
        plt.show()
            
        accuracy_matrix[f,:] = acc
    print('Feature x Pattern Accuracy matrix complete')
    return accuracy_matrix

# Plot sorted accuracy matrix mean errors
def sort_mat_err(mat):
    mean_errs = 1 - np.sort(np.mean(mat,1))[::-1]
    return mean_errs

# Generate entire accuracy matrix for some pseudo labels N times, return means
def gen_pseudo_mat(num_chems,N,num_msgs,num_traces,labels,feat_mat_norm):

    pseudo_acc_means = {}
    for k in range(N):
        print('Developing pseudo-accuracies for label set #',k,'/',N-1)
        
        # Perform a pseudo random label bootstrap, first develop the random labels
        pseudo_labels = {}
        for i in range(num_chems):
            pseudo_labels[i] = 1 + np.random.choice(int(num_msgs),int(num_traces[i]))
        
        # Applt the pseudo labels to develop an accuracy matrix, and mean vector
        pseudo_acc_mat = gen_acc_matrix(num_traces,pseudo_labels,feat_mat_norm,0)
        pseudo_acc_means[k] = sort_mat_err(pseudo_acc_mat)
        
    return pseudo_acc_means

def MDS_plot(feat_mat_norm,labels,num_msgs,c_inds):

    arr1 = feat_mat_norm[0][c_inds,:] # c_inds in first
    arr2 = feat_mat_norm[1][c_inds,:]
    concat_arr = np.concatenate((arr1,arr2),1)
    concat_arr = np.transpose(concat_arr)
    
    embedding = MDS(n_components=2)
    
    arr_transformed = embedding.fit_transform(concat_arr)
    
    for k in range(num_msgs-1):
        pat = k + 1
        loc1 = np.where(labels[0]==pat)[0]
        loc2 = len(labels[0]) + np.where(labels[1]==pat)[0]
        plt.scatter(arr_transformed[loc1,0],arr_transformed[loc1,1],marker='s')
        plt.scatter(arr_transformed[loc2,0],arr_transformed[loc2,1],marker='^')
    
#    plt.ylim((-2,2))
#    plt.xlim((-2,2))
    plt.title('2-component reduction of the feature dataset')
    plt.show()


# Generate an accuracy for every feature and every pattern
def mult_acc_matrix(num_traces,labels,feat_mat_norm,plot_feat_num):

    num_feats = np.shape(feat_mat_norm[0])[0] # just define these internally
    num_msgs = len(np.unique(labels[0]))
    acc = np.zeros(num_msgs)
    accuracy_matrix = np.zeros((num_feats,num_msgs))
    
    for f in range(num_feats):
        
        if f == round(num_feats/2):
            print('Halfway through features... (', f,'/',num_feats,')')
        
        for trace_num in range(num_msgs):
            
            # Seperate out features by chemical, plot a mxn grid for mxn traces
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
            
            acc[trace_num] = knn1d(num_obs,vals_feat,label_feat)
           
        plt.show()
            
        accuracy_matrix[f,:] = acc
    print('Feature x Pattern Accuracy matrix complete')
    return mult_acc_matrix


def append_pseudo_chem_data(num_pseudo_traces,feat_mat_norm,labels):
    key_lim = len(feat_mat_norm.keys())
    

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
        mpl.rcParams['figure.figsize']   = (20, 20)
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
        
        