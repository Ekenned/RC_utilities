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
#from sklearn.datasets import load_digits
#from sklearn.manifold import MDS
#from sklearn.neighbors import KNeighborsClassifier  

def load_TS_feat_matfile(wdir,fname,normalize):
    
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
    
    # And some ancillary information for normalization
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
    
    if normalize == 1:
        
        for chem_num in range(num_chems):
            
            for f in range(num_feats):
                all_trace_vals = feat_mat[chem_num][f,:]
                all_trace_vals = (all_trace_vals - abs_min[f]) / range_feat[f]
                feat_mat_norm[chem_num][f,:] = all_trace_vals 
    
    return chem_name,num_chems,num_feats,num_traces,num_msgs,labels,feat_mat_norm

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
        mpl.rcParams['figure.figsize']   = (22, 22)
        mpl.rcParams['axes.titlesize']   = 30
        mpl.rcParams['axes.labelsize']   = 30
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 20
        mpl.rcParams['xtick.labelsize']  = 30
        mpl.rcParams['ytick.labelsize']  = 30
    else:
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['figure.figsize']   = (10, 10)
        mpl.rcParams['axes.titlesize']   = 20
        mpl.rcParams['axes.labelsize']   = 20
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['xtick.labelsize']  = 16
        mpl.rcParams['ytick.labelsize']  = 16
        
    print('Modules and settings loaded')
        
        