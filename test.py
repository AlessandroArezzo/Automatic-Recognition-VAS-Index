from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
import argparse
import numpy as np
import os
import pandas as pd

"""Script that deals with performing two types of testing. 
 The first typology (type_test = 0) allows to compare the maximum scores obtained by varying the number of kernels (clusters)
 used to cluster the dataset in the preliminary clustering. Using fixed thresholds neutral configurations.
 The other typology (type_test = 1) allows to compare the maximum scores obtained by varying the threshold 
 that differentiates the relevant configurations for the classification of the vas index from the neutral ones during 
 the preliminary clustering. Using fixed number of clusters.
 The resulting score are saved in csv files. """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-type_test', "--type_test",
                        help="Determines type of tests: 0 for compare performance with differents n_kernels "
                             "in the preliminary clustering"
                             "and 1 for compare performance with differents thresholds neutral configuration "
                             "in the preliminary clustering",
                        default=0, type=int)
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=100, type=int)
    parser.add_argument('-threshold_neutral', "--threshold_neutral",
                        help="Threshold for neutral configuration in preliminary clustering",
                        default=0.03, type=float)
    parser.add_argument('-type_classifier', "--type_classifier",
                        help="Determines type of classifier to use: 'SVM' or 'SVR",
                        default='SVM')
    return parser.parse_args()


args = get_args()
type_test = args.type_test
# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
num_lndks = 66
percent_training_set = 0.85
# Features info
selected_lndks_idx = np.arange(66)
num_test_videos = 200
# Preliminary clustering info and paths
threshold_neutral = args.threshold_neutral # For type_test = 0
n_kernels_preliminary_clustering = args.n_kernels_preliminary_clustering  # For type_test = 1
# Model classifier info and paths
type_classifier = args.type_classifier
scores_result_thresholds_path ="data/test/" + str(n_kernels_preliminary_clustering)+"_kernels" \
                               + "/test_thresholds_"+type_classifier+"/scores_thresholds.csv"
scores_result_kernels_path = "data/test/test_n_kernels/score_results_"+type_classifier+"/scores_n_kernels.csv"

"""Check if all files and all directory passed as parameters existing"""


def check_existing_dump_paths(files_paths=[]):
    assert all(os.path.isfile(file_path) for file_path in files_paths)


"""The procedure is performed which involves performing preliminary clustering and subsequent generation 
of the classifier (SVM or SVR) given the number of kernels of the GMM and the threshold for the neutral configurations
to use in the preliminary clustering"""


def generate_and_test_model(n_kernels_preliminary_clustering, threshold_neutral_configurations,
                            preliminary_clustering=None):
    assert n_kernels_preliminary_clustering > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR') \
           and 0 < threshold_neutral_configurations < 1
    print("Experiments for #kernels=" + str(n_kernels_preliminary_clustering)+ " and threshold neutral configurations="
          +str(threshold_neutral_configurations))
    print("-- Preliminary clustering for #kernels=" + str(n_kernels_preliminary_clustering)+ " and threshold neutral configurations="
          +str(threshold_neutral_configurations))
    if preliminary_clustering == None:
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       num_test_videos=num_test_videos,
                                                       n_kernels=n_kernels_preliminary_clustering)
    preliminary_clustering.execute_preliminary_clustering(threshold_neutral=threshold_neutral_configurations)
    model_classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 num_test_videos=num_test_videos, preliminary_clustering=preliminary_clustering)
    model_classifier.train_model(percent_training_set=percent_training_set, train_by_max_score=True)
    score = model_classifier.calculate_rate_model(percent_data_set=1 - percent_training_set)
    return score, model_classifier.classifier.C, model_classifier.classifier.gamma

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the kernel number of the preliminary clustering gmm.
Save the results in a csv file containing the comparison of the best scores found for each threshold """

def compare_performance_different_thresholds():
    thresholds_neutral_to_test = np.arange(0.015, 0.05, 0.005)
    print("Prepare preliminary clustering...")
    preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                   seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                   selected_lndks_idx=selected_lndks_idx,
                                                   num_test_videos=num_test_videos,
                                                   n_kernels=n_kernels_preliminary_clustering)
    preliminary_clustering.execute_preliminary_clustering(threshold_neutral=0.015)
    out_df_scores = pd.DataFrame(columns=['thresholds_neutral', 'regularization_parameter', 'gamma_parameter', 'max_score'])
    max_score = optimal_thresholds = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for threshold in thresholds_neutral_to_test:
        score, regularization_parameter, gamma_parameter = generate_and_test_model(
            n_kernels_preliminary_clustering=n_kernels_preliminary_clustering,
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

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the threshold number 
for the preliminary clustering.
Save the results in a csv file containing the comparison of the best scores found for each number of kernels """


def compare_performance_different_number_clusters():
    n_kernels_to_test = np.arange(100, 200, 50)
    out_df_scores = pd.DataFrame(columns=['n_kernels', 'optimal_regularization', 'optimal_gamma', 'max_score'])
    max_score = optimal_n_kernels = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for n_kernels in n_kernels_to_test:
        score, regularization_parameter, gamma_parameter = generate_and_test_model(
            n_kernels_preliminary_clustering=n_kernels, threshold_neutral_configurations=threshold_neutral)
        data = np.hstack((np.array([n_kernels, regularization_parameter, gamma_parameter, score]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
        out_df_scores.to_csv(scores_result_kernels_path, index=False, header=True)
        if score > max_score:
            max_score = score
            optimal_n_kernels = n_kernels
            optimal_regularization_parameter = regularization_parameter
            optimal_gamma_parameter = gamma_parameter
    return optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter


if __name__ == '__main__':
    type_tests = args.type_test
    assert type_tests == 0 or type_tests == 1
    check_existing_dump_paths(files_paths=[coord_df_path, seq_df_path])
    if type_tests == 0:
        optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter = \
            compare_performance_different_number_clusters()
        print("End tests - Max rate with #kernels: " + str(optimal_n_kernels) +
              " and classifier with C= " + str(optimal_regularization_parameter) + " and gamma= "
              + str(optimal_gamma_parameter))
    elif type_tests == 1:
        max_rate, optimal_regularization_parameter, optimal_gamma_parameter = compare_performance_different_thresholds()
        print("End test with n_kernels= " + str(args.n_kernels_preliminary_clustering) + " -- Max rate= " + str(
            max_rate) + " - Optimal_C= " + str(optimal_regularization_parameter) + " and Optimal_gamma= " + str(
            optimal_gamma_parameter))
    else:
        exit(1)
