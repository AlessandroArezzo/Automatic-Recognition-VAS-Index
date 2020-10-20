import numpy as np
"""Configuration file for the test.py script that allows you to compare the maximum scores obtained by varying the threshold 
 that differentiates the relevant configurations for the classification of the vas index from the neutral ones during 
 the preliminary clustering. Using fixed number of clusters.
 The resulting score are saved in csv files."""

n_kernels_GMM = 50   # Number of clusters of the GMM ( to use only for type_test=1)

selected_lndks_idx = [5, 11, 19, 24, 37, 41, 56, 58] # Indexes of the landmarks to use for fitting GMM and description sequences

type_classifier = 'SVR'  # Indicates the type of the classifier ('SVR' or 'SVM')

thresholds_neutral_to_test = np.arange(0.015, 0.06, 0.005)  #  Thresholds to test (only for type_test=1)

cross_val_protocol = "5-fold-cross-validation"  # Define type of protocol to be used to evaluate the performance of the models
"""cross_val_protocol:  'Leave-One-Sequence-Out' or '5-fold-cross-validation' """