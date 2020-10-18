import math

from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
from configuration import config_tests
import numpy as np
import os
import pandas as pd

def check_existing_paths(dir_paths=[],file_paths=[]):
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            print("Configuration error: dir path '"+dir_path+"' not exist in project")
            exit(1)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print("Configuration error: file '" + file_path + "' not exist in project")
            exit(1)

# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
num_lndks = 66
percent_training_set = 0.85
# Features info
selected_lndks_idx = config_tests.selected_lndks_idx
num_test_videos = 200
train_video_idx = np.arange(0,num_test_videos)[:int(percent_training_set * num_test_videos)]
test_video_idx = np.arange(0,num_test_videos)[int(percent_training_set * num_test_videos):num_test_videos]
# Preliminary clustering info and paths
n_kernels_GMM = config_tests.n_kernels_GMM
thresholds_neutral_to_test = config_tests.thresholds_neutral_to_test
# Model classifier info and paths
type_classifier = config_tests.type_classifier
scores_result_thresholds_path ="data/test/" + str(n_kernels_GMM)+"_kernels" \
                               + "/test_thresholds_"+type_classifier+"/scores_thresholds.csv"
scores_result_kernels_path = "data/test/test_n_kernels/score_results_"+type_classifier+"/scores_n_kernels.csv"

"""The procedure is performed which involves performing preliminary clustering and subsequent generation 
of the classifier (SVM or SVR) given the number of kernels of the GMM and the threshold for the neutral configurations
to use in the preliminary clustering"""


def generate_and_test_model(n_kernels_GMM, threshold_neutral_configurations,
                            preliminary_clustering):
    assert n_kernels_GMM > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR') \
           and 0 < threshold_neutral_configurations < 1
    model_classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 train_video_idx=train_video_idx, test_video_idx=test_video_idx,
                                preliminary_clustering=preliminary_clustering)
    model_classifier.train_model(train_by_max_score=True)
    score = model_classifier.calculate_rate_model()
    return score, model_classifier.classifier.C, model_classifier.classifier.gamma

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the kernel number of the preliminary clustering gmm.
Save the results in a csv file containing the comparison of the best scores found for each threshold """

def compare_performance_different_thresholds():
    preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                   seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                   selected_lndks_idx=selected_lndks_idx,
                                                   train_video_idx=train_video_idx,
                                                   n_kernels=n_kernels_GMM)
    out_df_scores = pd.DataFrame(columns=['thresholds_neutral', 'regularization_parameter', 'gamma_parameter', 'max_score'])
    max_score = optimal_thresholds = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for threshold_idx in np.arange(0,len(thresholds_neutral_to_test)):
        threshold = thresholds_neutral_to_test[threshold_idx]
        threshold = round(threshold, 5 - int(math.floor(math.log10(abs(threshold)))) - 1)
        print("Execute experiments using threshold=" + str(threshold) + "...")
        preliminary_clustering.execute_preliminary_clustering(threshold_neutral=threshold)
        if len(preliminary_clustering.index_relevant_configurations) == 0:
            score = regularization_parameter = gamma_parameter = "None"
        else:
            score, regularization_parameter, gamma_parameter = generate_and_test_model(
                n_kernels_GMM=n_kernels_GMM,
                threshold_neutral_configurations=threshold, preliminary_clustering=preliminary_clustering)
        data = np.hstack((np.array([threshold, regularization_parameter, gamma_parameter, score]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
        out_df_scores.to_csv(scores_result_thresholds_path, index=False, header=True)
        if score > max_score:
            max_score = score
            optimal_thresholds = threshold
            optimal_regularization_parameter = regularization_parameter
            optimal_gamma_parameter = gamma_parameter
    return optimal_thresholds, optimal_regularization_parameter, optimal_gamma_parameter

if __name__ == '__main__':
    dir_paths = ["data/test/" + str(n_kernels_GMM)+"_kernels" + "/test_thresholds_"+type_classifier+"/"]
    file_paths = [coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    print("Execute tests with different thresholds for the neutral configurations (using "+str(n_kernels_GMM)+" kernels)")
    max_threshold, optimal_regularization_parameter, optimal_gamma_parameter = compare_performance_different_thresholds()
    print("End test with n_kernels= " + str(n_kernels_GMM) + ": best threshold= " + str(
        max_threshold) + " with Optimal_C= " + str(optimal_regularization_parameter) + " and Optimal_gamma= " + str(
        optimal_gamma_parameter))
