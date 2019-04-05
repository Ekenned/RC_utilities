# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:57:32 2019

@author: Eamonn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import TS_sub_functions # Custom time series sub-functions and processes
from TS_class import *
 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# User inputs:

wdir    = r"C:\Users\Eamonn\Documents\GitHub\RC_utilities\Compare_TS_feats\example_feat_data"
fname   = r'chem_ts_feat'
norm    = 0 # Set to 1 to normalize all data by feature range to [0,1]
pl_num  = 1 # increase this to plot more graphs, or 0 for no func plots
thresh  = 0.99 # Set required accuracy for a feature to be considered 'useful'
N       = 2 # Create pseudo accuracy matrices for bootstrapping

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

load_settings()

inst_a = TSFA(wdir,fname)  

inst_a.change_directory()
 
inst_a.load_mat_file()

inst_a.define_dicts()

inst_a.get_info()

inst_a.get_arrays()

inst_a.mult_acc_matrix()

plt.loglog(sort_mat_err(inst_a.accuracy_matrix),'.')
plt.xlabel('Sorted feature')
plt.ylabel('Average feature error')
plt.ylim((.001,1))
plt.xlim((1,1000))
plt.grid()