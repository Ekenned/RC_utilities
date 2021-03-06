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

class TSFA():
    def __init__(self,wdir,fname):
        self.wdir = wdir
        self.fname = fname
        self.norm = 0 # default do not normalize feature ranges
        
    def change_directory(self):
        os.chdir(self.wdir)
        print(os.getcwd())
        
    def load_mat_file(self):
        print('Loading from', self.fname + '.mat')
        mat_data = sio.loadmat(self.fname + '.mat')
        self.data = mat_data[self.fname]
        # return mat_data
        
    def define_dicts(self):
        self.label_dict = self.data['labels'][0][0]
        self.feat_dict = self.data['feat_mat'][0,0]
        self.names = self.data['chem_names'][0,0]
        self.n_chems = np.shape(self.names)[0]
        self.n_feats = np.shape(self.feat_dict[0][self.names[0][0][0]][0])[0]
        
    def get_info(self):
        chem_names = {}
        print('Chemicals used:')
        for j in range(self.n_chems):
            chem_names[j] = self.names[j][0][0]
            print(chem_names[j])
        print('# Features:', self.n_feats)
        self.chem_names = chem_names

    def get_arrays(self):
    # def get_mfile_arrays(num_feats,chem_names,feat_dict,label_dict,norm):
        
        # Core information, pattern labels, features, chemicals 
        labels = {}
        feat_mat = {}
        chem_name = {}
        nc = self.n_chems
        nf = self.n_feats
        
        # And some ancillary information for normalization
        n_traces = np.zeros(nc)
        feat_max_arr = np.zeros((nf,nc))
        feat_min_arr = np.zeros((nf,nc))
        
        # Get the labels and feature matrix for every chemical
        for c in range(nc):
        
            feat_mat[c] = self.feat_dict[self.chem_names[c]][0,0]
            labels[c] = self.label_dict[self.chem_names[c]][0][0]
            feat_mat[c][np.isnan(feat_mat[c])] = 0 # remove nan entries
            n_traces[c] = np.shape(feat_mat[c])[1]
            feat_max_arr[:,c] = np.max(feat_mat[c],1)
            feat_min_arr[:,c] = np.min(feat_mat[c],1)
            self.n_msgs = len(np.unique(labels[0]))    
    
        # Normalize data from 0 - 1 
        abs_max = np.nanmax(feat_max_arr,1) # max and min over all chemicals by feature
        abs_min = np.nanmin(feat_min_arr,1)
        range_feat = abs_max - abs_min + 10**-6
        feat_mat_norm = feat_mat # instantiate the normalized vector
        
        if self.norm == 1:
            
            for c in range(nc):
                
                for f in range(self.n_feats):
                    all_trace_vals = feat_mat[c][f,:]
                    all_trace_vals = (all_trace_vals - abs_min[f]) / range_feat[f]
                    feat_mat_norm[c][f,:] = all_trace_vals 
                    
        self.n_traces = n_traces
        self.feat_mat_norm = feat_mat_norm
        self.labels = labels
        
        print('# obs per chemical: ', n_traces)
        
    def return_feat_mat(self):
        return self.feat_mat_norm 
    
    def return_labels(self):
        return self.labels      

    # Generate an accuracy for every feature and pattern for N chemicals
    #def mult_acc_matrix(num_traces,labels,feat_mat_norm,plot_feat_num):   
    def mult_acc_matrix(self):
        
        plot_feat_num = 1 # Just plot 1 features
        nf = self.n_feats # just define these internally
        n_msgs = self.n_msgs
        acc = np.zeros(n_msgs)
        accuracy_matrix = np.zeros((nf,n_msgs))
        n_chems = self.n_chems
        
        for f in range(nf):
            
            if f == round(nf/2):
                print('Halfway through features... (', f,'/',nf,')')
            
            for msg in range(n_msgs):
                
                # Seperate out features by chemical
                v = {}
                num_obs = np.zeros(n_chems)
                vals_feat = np.array([])
                label_feat = np.array([])
                for c in range(n_chems):
                    v[c] = self.feat_mat_norm[c][f,np.where(self.labels[c]==msg+1)[0]]
                    num_obs[c] = len(v[c])
                    vals_feat = np.append(vals_feat,v[c])
                    label_feat = np.append(label_feat,c*(np.ones(int(num_obs[c]))))
                net_obs = sum(num_obs)
    
                if f<plot_feat_num: 
                    x,y = get_best_subplot(n_msgs)
                    plt.subplot(y,x,msg + 1)
                    # plt.scatter(label_feat,vals_feat,c=label_feat)
                    plt.scatter(range(len(vals_feat)),vals_feat,c=label_feat,vmin=-.5,vmax=1.5)
                    plt.title(msg) 
                    # plt.ylim((0,1))
                    plt.yticks([])
                
                acc[msg] = knn1d(int(net_obs),vals_feat,label_feat)
               
            plt.show()
                
            accuracy_matrix[f,:] = acc
        # print('Feature x Pattern Accuracy matrix complete')
        self.accuracy_matrix = accuracy_matrix
        
# Iterate 1D 1NN over all points and test against the true labels
def knn1d(num_obs,vals,labels):
    label_1nn = np.zeros(num_obs)
    for i in range(num_obs):
        distances = np.abs(vals - vals[i])
        distances[i] = np.max(distances) + 1 # rule out the value itself
        label_1nn[i] = labels[np.where(distances==np.min(distances))[0][0]]
        sing_acc = np.sum(1*(labels==label_1nn))/num_obs
    return sing_acc 

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
        pseudo_acc_mat = mult_acc_matrix(num_traces,pseudo_labels,feat_mat_norm,0)
        pseudo_acc_means[k] = sort_mat_err(pseudo_acc_mat)
        
    return pseudo_acc_means

# Concatenate all chemical data, reduce matrix to features of interest
def concat_sub_feats(feat_mat_norm,labels,c_inds):
    
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

# Get reduced multidimensional representation of the data
def MDS_plot(feat_mat_norm,labels,c_inds):
    
    label_feat,concat_arr = concat_sub_feats(feat_mat_norm,labels,c_inds)
    
    embedding = MDS(n_components=2)
    
    arr_trans = embedding.fit_transform(concat_arr)
    
    marker_list =	{0: ".",1: "o",2: "^",3: "s",4: "x"}
    for c in range(len(feat_mat_norm.keys())):
        inds = np.where(label_feat==c)[0]
        plt.scatter(arr_trans[inds,0],arr_trans[inds,1],marker=marker_list[c])
    plt.title('2-component reduction of discriminating feature')
    plt.show()
    
def knn_mult(train_samp,concat_arr,label_feat,n):
    
    # train_samp = 500; n = 3
    classifier = KNeighborsClassifier(n_neighbors = n)  
    ns = np.shape(concat_arr)
    
    train_lbls = np.random.choice(ns[0], size = train_samp, replace=False)
    test_lbls = np.setdiff1d(range(ns[0]),train_lbls)
    
    classifier.fit(concat_arr[train_lbls,:], label_feat[train_lbls]) 
    y = classifier.predict(concat_arr[test_lbls,:])
    # y_net = classifier.predict(concat_arr[test_lbls,:])
    acc_test = np.sum(y == label_feat[test_lbls]) / (ns[0] - train_samp)
    # acc_net = np.sum(y_net == label_feat) / ns[0]
    return acc_test # ,acc_net

# Repeat multifeature knn N times
def rep_knn_mult(num_reps,train_samp,concat_arr,label_feat,n):
    # num_reps = 100; train_samp = 400 ; n = 1;
    acc_tests = np.zeros(num_reps)
    for i in range(num_reps):
        acc_tests[i] = knn_mult(train_samp,concat_arr,label_feat,n)
    print('Classification error (', train_samp,'/',len(label_feat), ' test traces): ',1 - np.median(acc_tests))
    print(' \n ')
    return acc_tests

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

        
        