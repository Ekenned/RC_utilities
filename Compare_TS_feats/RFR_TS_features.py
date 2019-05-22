# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:06:53 2019

@author: Eamonn
"""

from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from scipy.io import loadmat
from scipy.ndimage.filters import maximum_filter1d,median_filter
from scipy.ndimage.measurements import label
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

use_feats = range(n_feats)
#num_sens_feats = rec_f*n_vars
#use_feats = np.array([])
#for i in range(2*num_sens-1):
#    use_feats = np.append(use_feats,i*num_sens_feats + np.array([range(rec_f,rec_f*3)]))
#np.remainder(use_feats,rec_f)
num_trials = 20
err_frac = np.zeros(num_trials)
for i in range(num_trials):
    train_split = 0.8
    
    label_feat,concat_arr = concat_sub_feats(feat_mat,use_feats)
    n_obs = len(label_feat)
    
    num_train       = round(train_split*n_obs)
    num_test        = n_obs - num_train   
    train_indices   = np.random.choice(n_obs, num_train, replace=False)
    train_matrix    = concat_arr[train_indices,:]
    train_labels    = label_feat[train_indices]
    test_indices    = np.setdiff1d(range(n_obs),train_indices)
    test_matrix     = concat_arr[test_indices,:]
    test_labels     = label_feat[test_indices]
    
    regr = RandomForestRegressor(max_features=2000,max_depth=None, random_state=0,n_estimators=500)
    
    regr.fit(concat_arr[train_indices,:],train_labels.astype(int))
    
    feat_pow = regr.feature_importances_
    
    predict_labels = regr.predict(concat_arr)
    
    num_errs = np.sum(np.abs(np.round(predict_labels) - label_feat))
    
    err_frac[i] = num_errs/n_obs
    print(err_frac[i])
    
print(np.mean(err_frac))
plt.plot(label_feat)
plt.plot(predict_labels)
plt.show()

plt.stem(np.round(predict_labels) - label_feat)



