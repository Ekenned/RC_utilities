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

num_trials = 100
train_split = 0.5
acc_lim = 0.5 # only use features with accuracy above some value

use_feats = np.where(accuracy_vec>acc_lim)[0]


label_feat,concat_arr = concat_sub_feats(feat_mat,use_feats)
concat_arr = zero_nan_and_infs(concat_arr)
n_obs = len(label_feat)
    
err_frac = np.zeros(num_trials)
for i in range(num_trials):
    
    # split the matrix and labels into test and train subsections
    train_inds,test_inds = test_train_inds(n_obs,train_split)
    
    train_matrix    = concat_arr[train_inds,:]
    train_labels    = label_feat[train_inds]
    test_matrix     = concat_arr[test_inds,:]
    test_labels     = label_feat[test_inds]
    
    regr = RandomForestRegressor(max_features=1000,max_depth=None, random_state=0,n_estimators=200)
    
    model = regr.fit(train_matrix,train_labels)
    
    feat_pow = model.feature_importances_
    
    predict_labels = model.predict(test_matrix)
    
    num_errs = np.sum(np.abs(np.round(predict_labels) - test_labels))
    
    err_frac[i] = num_errs/len(test_labels)
    print(err_frac[i])
    
    if err_frac[i]<=0.03:
        print('Good example')
        keep_test_labels = test_labels
        keep_predict_labels = predict_labels
        break
    
#    plt.plot(test_labels)
#    plt.plot(predict_labels,'o')
#    plt.show()
    
print(np.mean(err_frac))
plt.plot(test_labels)
plt.plot(predict_labels,'o')
plt.show()

plt.stem(1*((np.round(predict_labels) - test_labels)>0))
plt.show()

feat_lim = 0.001
X = np.remainder(use_feats[np.where(feat_pow>feat_lim)[0]],rec_f)
Y = feat_pow[np.where(feat_pow>feat_lim)[0]]

sum_Y = np.zeros(np.max(X))
for i in range(0,np.max(X)):
    loc_inds = np.where(X==i)[0]
    sum_Y[i] = np.sum(Y[loc_inds])
n_bins = 40    
max_val = 2.5
plt.hist(predict_labels[test_labels==2],bins=n_bins,range=((0,max_val)))
plt.hist(predict_labels[test_labels==1],bins=n_bins,range=((0,max_val)))
plt.hist(predict_labels[test_labels==0],bins=n_bins,range=((0,max_val)))
plt.ylim((0,10))
plt.savefig('fig_pts.eps', format='eps', dpi=1000)


