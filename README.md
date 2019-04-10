# RC_utilities
Utilities for RC project. This may end up encompassing more then trufflebot.

'Classify_TS_features.py' is a script to read in time series features and perform single feature classification of EtOH and Acetone

'TS_sub_functions.py' and 'TS_class.py' are the underlying function/class lists for data structuring, labeling, knn, and plots.

The example data (chem_ts_feat.mat) used by the classifier is in the folder ...\RC_utilities\Compare_TS_feats\example_feat_data

This dataset is  comprised of labels (1-24) and time series features (6x219) gathered from different patterns (24) and chemicals (2)

'Compare_TS_feats' folder will deal with scripts for comparing time series features across chemical, for identification.
