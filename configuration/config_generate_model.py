import numpy as np
"""Configuration file for the generate_model_predictor.py script that allows you to generate
an SVR predictor from its parameters"""

# Number of clusters of the GMM
n_kernels_GMM = 200

# Threshold of the neutral configurations
threshold_neutral = None
""" For example if the threshold is 0.3: all those configurations that occur within the sequences with vas equal to 0 
 with a frequency greater than 0.3 will be considered neutral.
  If threshold is None: all those configurations of the sequences with vas equal to 0 that
 have a frequency equal to half the maximum detected in all the videos are considered neutral. """

# Indexes of the landmarks to use for fitting GMM and description sequences
selected_lndks_idx = [5, 11, 19, 24, 37, 41, 56, 58]

# Defines if the histograms of the dataset sequences must be saved in their respective files
save_histo_figures = False
"""If save_histo_figures = True, the histograms are saved in the project folder
 'data/classifier/n_kernels/figures/histograms/' with n=number of kernels of GMM
 (make sure that this file exists)"""

# Type of protocol to be used to evaluate the performance of the models
cross_val_protocol = "5-fold-cross-validation"
"""cross_val_protocol:  'Leave-One-Sequence-Out' or '5-fold-cross-validation' """
