from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
import argparse
import numpy as np
import os
import pandas as pd

"""Script that deals with performing two types of testing. 
 The first typology (type_test = 0) allows to obtain the regularization and gamma parameters for the classifier (SVM or SVR)
 which minimize the error on the training data using a specific number of clusters for the preliminary phase. 
 The other typology (type_test = 1) allows to compare the maximum scores obtained by varying the number of kernels (clusters)
 used to cluster the dataset in the preliminary clustering.
 The resulting score are saved in csv files. """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-type_test', "--type_test",
                        help="Determines type of tests: 0 for score parameter classifier with select n_kernels_preliminary_clustering"
                             "and 1 for compare performance with differents n_kernels for preliminary clustering",
                        default=0, type=int)
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=100, type=int)
    parser.add_argument('-threshold_neutral', "--threshold_neutral",
                        help="Threshold for neutral configuration in preliminary clustering",
                        default=0.3, type=float)
    parser.add_argument('-threshold_relevant', "--threshold_relevant",
                        help="Threshold for relevant configuration in preliminary clustering",
                        default=0.2, type=float)
    parser.add_argument('-save_histo_figures', "--save_histo_figures",
                        help="Determines if histograms are to be saved during preliminary clustering phases",
                        default=False, type=bool)
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
threshold_neutral = args.threshold_neutral
threshold_relevant = args.threshold_relevant
n_kernels_preliminary_clustering = args.n_kernels_preliminary_clustering  # For type_test = 0
save_histo_figures = args.save_histo_figures # For type_test = 0
histo_figures_path = "data/test/"+str(n_kernels_preliminary_clustering)+"_kernels/figures/histograms" # For save_histo_figures = True
# Model classifier info and paths
type_classifier = args.type_classifier
rating_parameters_path = "data/test/" + str(n_kernels_preliminary_clustering)+"_kernels" + "/test_" + type_classifier + "_parameters/rate_classifiers"
scores_result_kernels_path = "data/test/test_n_kernels/score_results/scores_number_kernels.csv"

"""Check if all files and all directory passed as parameters existing"""


def check_existing_dump_paths(files_paths=[], dir_paths=[]):
    assert all(os.path.isfile(file_path) for file_path in files_paths)
    assert all(os.path.isdir(dir_path) for dir_path in dir_paths)


"""Given a specific number of kernels for the preliminary clustering, 
compare the scores obtained by varying the gamma and regularization parameters used for the classifier (SVM or SVR).
Save the results in csv files """


def test_best_classifier_parameters(n_kernels, print_score_result=True, plot_and_save_histo=False):
    type_classifier = args.type_classifier
    threshold_neutral = args.threshold_neutral
    threshold_relevant = args.threshold_relevant
    assert n_kernels > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR') and 0 < threshold_neutral < 1 and \
           0 < threshold_relevant < 1
    print("Experiments for #kernels= " + str(n_kernels))
    print("-- Preliminary clustering for #kernels= " + str(n_kernels))
    preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                   seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                   selected_lndks_idx=selected_lndks_idx,
                                                   num_test_videos=num_test_videos,
                                                   n_kernels=n_kernels, threshold_neutral=threshold_neutral,
                                                   threshold_relevant=threshold_relevant)
    preliminary_clustering.execute_preliminary_clustering(histo_figures_path=histo_figures_path,
                                                          plot_and_save_histo=plot_and_save_histo)
    regularization_test_parameters = np.arange(10, 1010, 10)
    gamma_test_parameters = np.arange(0.1, 1.1, 0.1)
    max_rate = optimal_regularization_parameter = optimal_gamma_parameter = 0
    classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 num_test_videos=num_test_videos,preliminary_clustering=preliminary_clustering)
    for regularization in regularization_test_parameters:
        for gamma in gamma_test_parameters:
            print("-- Train model classifier for #kernels= " + str(n_kernels) + ' with C= ' + str(regularization) +
                  ' - gamma= ' + str(gamma))
            classifier.train_model(percent_training_set=percent_training_set,
                                   regularization_parameter=regularization,
                                   gamma_parameter=gamma)
            if print_score_result == True:
                score_path = rating_parameters_path + '/' + str(regularization) + '_' + str(gamma) + '.csv'
            current_rate = classifier.calculate_rate_model(percent_data_set=1 - percent_training_set,
                                                           path_scores_parameters=score_path)
            if current_rate > max_rate:
                max_rate = current_rate
                optimal_regularization_parameter = regularization
                optimal_gamma_parameter = gamma
    return max_rate, optimal_regularization_parameter, optimal_gamma_parameter

"""Compare the best scores obtained by varying the gamma and regularization parameters of the classifier 
according to the number of kernels used for the preliminary clustering. 
Save the results in a csv file containing the comparison of the best scores found for each number of kernels """


def compare_performance_different_number_clusters():
    n_kernels_to_test = np.arange(100, 200, 50)
    out_df_scores = pd.DataFrame(columns=['n_kernels', 'optimal_regularization', 'optimal_gamma', 'max_rate'])
    max_rate = optimal_n_kernels = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for n_kernels in n_kernels_to_test:
        rate, regularization_parameter, gamma_parameter = test_best_classifier_parameters(n_kernels=n_kernels,
                                                                                          print_score_result=False)
        data = np.hstack((np.array([n_kernels, regularization_parameter, gamma_parameter, rate]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                             ignore_index=True)
        out_df_scores.to_csv(scores_result_kernels_path, index=False, header=True)
        if rate > max_rate:
            max_rate = rate
            optimal_n_kernels = n_kernels
            optimal_regularization_parameter = regularization_parameter
            optimal_gamma_parameter = gamma_parameter
    return optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter


if __name__ == '__main__':
    type_tests = args.type_test
    assert type_tests == 0 or type_tests == 1
    if type_tests == 0:
        check_existing_dump_paths(files_paths=[coord_df_path, seq_df_path],
                                  dir_paths=[histo_figures_path, rating_parameters_path])
        max_rate, optimal_regularization_parameter, optimal_gamma_parameter = test_best_classifier_parameters(
                                                                        n_kernels=n_kernels_preliminary_clustering)
        print("End test with n_kernels= " + str(args.n_kernels_preliminary_clustering) + " -- Max rate= " + str(
            max_rate) +
              " - Optimal_C= " + str(optimal_regularization_parameter) + " and Optimal_gamma= " + str(
            optimal_gamma_parameter))
    else:
        check_existing_dump_paths(files_paths=[coord_df_path, seq_df_path])
        optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter = \
            compare_performance_different_number_clusters()
        print("End tests - Max rate with #kernels: " + str(optimal_n_kernels) +
              " and classifier with C= " + str(optimal_regularization_parameter) + " and gamma= "
              + str(optimal_gamma_parameter))
