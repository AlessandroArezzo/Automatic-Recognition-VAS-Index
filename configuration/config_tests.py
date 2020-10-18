import numpy as np
"""Configuration file for the test.py script that allows you to compare the maximum scores obtained by varying the threshold 
 that differentiates the relevant configurations for the classification of the vas index from the neutral ones during 
 the preliminary clustering. Using fixed number of clusters.
 The resulting score are saved in csv files."""

n_kernels_GMM = 50   # Number of clusters of the GMM ( to use only for type_test=1)

selected_lndks_idx = np.arange(66)  # Indexes of the landmarks to use for fitting GMM and description sequences

type_classifier = 'SVR'  # Indicates the type of the classifier ('SVR' or 'SVM')

thresholds_neutral_to_test = np.arange(0.015, 0.06, 0.005)  #  Thresholds to test (only for type_test=1)