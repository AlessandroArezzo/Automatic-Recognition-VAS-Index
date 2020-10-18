import numpy as np
"""Configuration file for the test.py script that allows you to performs two types of testing. 
 The first typology (type_test = 0) allows to compare the maximum scores obtained by varying the number of kernels (clusters)
 used to cluster the dataset in the preliminary clustering. Using fixed thresholds neutral configurations.
 The other typology (type_test = 1) allows to compare the maximum scores obtained by varying the threshold 
 that differentiates the relevant configurations for the classification of the vas index from the neutral ones during 
 the preliminary clustering. Using fixed number of clusters.
 The resulting score are saved in csv files."""

type_test = 1  # Define typology of the tests (0 or 1)

threshold_neutral = 0.3  # # Defines the threshold of the neutral configurations (to use only for type_test=0)
""" If the threshold is 0.3, all those configurations that occur within the sequences with vas equal to 0 
 with a frequency greater than 0.3 will be considered neutral """

n_kernels_GMM = 100   # Number of clusters of the GMM ( to use only for type_test=1)

selected_lndks_idx = np.arange(66)  # Indexes of the landmarks to use for fitting GMM and description sequences

type_classifier = 'SVR'  # Indicates the type of the classifier ('SVR' or 'SVM')

n_kernels_to_test = np.arange(50, 800, 50)  #  Numbers of kernels to test (only for type_test=0)

thresholds_neutral_to_test = np.arange(0.015, 0.05, 0.005)  #  Thresholds to test (only for type_test=1)