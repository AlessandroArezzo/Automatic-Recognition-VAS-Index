import numpy as np
""" Configuration file """


# PARAMETERS USED BY ALL SCRIPTS

# Define if the GMM must be fitted minimized BIC from more than one kernels number
fit_by_bic = False

# Number of clusters of the GMM (if fit_by_bic = True set a list of number of kernels, otherwise set an integer value)
n_kernels_GMM = 16

# Covariance type to use for the GMM fitting
covariance_type = "full"
"""covariance_type: 'diag' or 'full' """

# Indexes of the landmarks to use for fitting GMM and description sequences
selected_lndks_idx = [5, 11, 19, 24, 37, 41, 44, 46, 50, 52, 56, 58]
#selected_lndks_idx = np.arange(66)

n_jobs = 4  # Number of threads to use to perform SVR training

# Type of protocol to be used to evaluate the performance of the models
cross_val_protocol = "5-fold-cross-validation"
"""cross_val_protocol:  'Leave-One-Subject-Out' or '5-fold-cross-validation' or 'Leave-One-Sequence-Out'"""



# PARAMETERS USED BY THE SCRIPT generate_model_predictor.py

# Threshold of the neutral configurations (if fit_by_bic = True set a list of thresholds of the same length defined
# in the n_kernels_GMM list, otherwise set a float value between 0 and 1)
threshold_neutral = 0.3
""" For example if the threshold is 0.3: all those configurations that occur within the sequences with vas equal to 0 
 with a frequency greater than 0.3 will be considered neutral. """

# Defines if the histograms of the dataset sequences must be saved in their respective files
save_histo_figures = False
"""If save_histo_figures = True, the histograms are saved in the project folder
 'data/classifier/n_kernels/figures/histograms/' with n=number of kernels of GMM
 (make sure that this file exists)"""

# Defines if the samples must be weighted for training
weighted_samples = True


# PARAMETERS USED BY THE SCRIPT test.py

# Thresholds to test
thresholds_neutral_to_test = np.arange(0.1, 0.6, 0.05)