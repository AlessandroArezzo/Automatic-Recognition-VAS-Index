import numpy as np
"""Configuration file for the test.py script that allows you to compare the maximum scores obtained by varying the threshold 
 that differentiates the relevant configurations for the classification of the vas index from the neutral ones during 
 the preliminary clustering. Using fixed number of clusters.
 The resulting score are saved in csv files."""

# Number of clusters of the GMM
n_kernels_GMM = 100

# Indexes of the landmarks to use for fitting GMM and description sequences
selected_lndks_idx = [5, 11, 19, 24, 37, 41, 56, 58]
#selected_lndks_idx = np.arange(66)

# Thresholds to test
thresholds_neutral_to_test = np.arange(0.01, 0.1, 0.005)

# Type of protocol to be used to evaluate the performance of the models
cross_val_protocol = "5-fold-cross-validation"
"""cross_val_protocol:  'Leave-One-Sequence-Out' or '5-fold-cross-validation' """