# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import osimport numpy as np
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
from sklearn.datasets import load_digitsfrom sklearn.manifold import MDS

wdir            = r"Z:\userdata\ekennedy\scripts\temp\sens1"
fname_EtOH = r'feature_struct.mat'
fname_Ace = r'feature_struct2.mat'

data = sio.loadmat(fname_EtOH)
EtOH_data = data['feature_struct']
EtOH_labels = EtOH_data['labels'][0,0]
EtOH_mat = EtOH_data['lv_norm'][0,0]

data = sio.loadmat(fname_Ace)
Ace_data = data['feature_struct2']
Ace_labels = Ace_data['labels'][0,0]
Ace_mat = Ace_data['lv_norm'][0,0]



