from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
import argparse
import numpy as np
import os
import glob
import pandas as pd

"""Script that deals with performing two types of testing. 
 The first typology (type_test = 0) allows to obtain the regularization and gamma parameters for the classifier (SVM or SVR)
 which minimize the error on the training data using a specific number of clusters for the preliminary phase. 
 The other typology (type_test = 1) allows to compare the maximum scores obtained by varying the number of kernels (clusters)
 used to cluster the dataset in the preliminary clustering.
 The resulting score are saved in csv files. """
def get_args():
    parser = argparse.ArgumentParser()
    # Dataset info
    parser.add_argument('-coord_df_path', "--coord_df_path", help="Path coordinates df",
                        default='data/dataset/2d_skeletal_data_unbc_coords.csv')
    parser.add_argument("-seq_df_path", "--seq_df_path", help="Path sequences df",
                        default='data/dataset/2d_skeletal_data_unbc_sequence.csv')
    parser.add_argument('-num_lndks', "--num_lndks", help="Number of facial landmarks", default=66, type=int)
    parser.add_argument('-percent_training_set', "--percent_training_set",
                        help="Percent of data to use for training set of the classifier model",
                        default=0.85, type=float)
    # Features info
    parser.add_argument('-selected_lndks_idx', "--selected_lndks_idx", help="Number of facial landmarks",
                        default=np.arange(66), type=int)
    parser.add_argument('-num_test_videos', "--num_test_videos", help="Number of videos to use for test",
                        default=200, type=int)
    # Clustering info
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=150, type=int)
    parser.add_argument('-threshold_neutral', "--threshold_neutral",
                        help="Threshold for neutral configuration in preliminary clustering",
                        default=0.3, type=float)
    parser.add_argument('-threshold_relevant', "--threshold_relevant",
                        help="Threshold for relevant configuration in preliminaryclustering",
                        default=0.2, type=float)
    parser.add_argument('-save_histo_figures', "--save_histo_figures",
                        help="Determines if histograms are to be saved during preliminary clustering phases",
                        default=False, type=bool)
    parser.add_argument('-type_classifier', "--type_classifier",
                        help="Determines type of classifier to use: 'SVM' or 'SVR",
                        default='SVM')
    #Test info
    parser.add_argument('-type_test', "--type_test",
                        help="Determines type of tests: 0 for score parameter classifier with select n_kernels_preliminary_clustering"
                             "and 1 for compare performance with differents n_kernels for preliminary clustering",
                        default=0, type=int)
    # Path files generated
    n_kernels = parser.parse_args().n_kernels_preliminary_clustering
    type_test = parser.parse_args().type_test
    if type_test == 0:
        sub_directory = str(n_kernels) + "_kernels"
    else:
        sub_directory='test_n_kernels'
    parser.add_argument('-histo_figures_path', "--histo_figures_path", help="Path histograms figures",
                        default='data/test/' + sub_directory + '/figures/histograms')
    parser.add_argument('-models_dump_path', "--models_dump_path",
                        help="Path for dump preliminary clustering and model classifier",
                        default='data/test/' + sub_directory)
    type_classifier = parser.parse_args().type_classifier
    parser.add_argument('-rating_parameters_path', "--rating_parameters_path",
                        help="Path rating classifiers parameters",
                        default='data/test/' + sub_directory + '/test_' + type_classifier + '_parameters/rate_classifiers')
    parser.add_argument('-scores_result_kernels_path', "--scores_result_kernels_path",
                        help="Path max rates with different number of kernels for preliminary clustering",
                        default='data/test/' + sub_directory + '/score_results')
    return parser.parse_args()

def check_existing_dump_paths(files_paths, dir_paths):
    assert all(os.path.isfile(file_path) for file_path in files_paths)
    assert all(os.path.isdir(dir_path) for dir_path in dir_paths)

def test_best_classifier_parameters(args, n_kernels, print_score_result = True, plot_and_save_histo=False):
    type_classifier = args.type_classifier
    threshold_neutral = args.threshold_neutral
    threshold_relevant = args.threshold_relevant
    assert n_kernels > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR') and 0 < threshold_neutral < 1 and \
           0 < threshold_relevant < 1
    print("Experiments for #kernels= " + str(n_kernels))
    print("-- Preliminary clustering for #kernels= " + str(n_kernels))
    preliminary_clustering = PreliminaryClustering()
    preliminary_clustering.execute_preliminary_clustering(coord_df_path=args.coord_df_path,
                                                          seq_df_path=args.seq_df_path, num_lndks=args.num_lndks,
                                                          selected_lndks_idx=args.selected_lndks_idx,
                                                          num_test_videos=args.num_test_videos,
                                                          n_kernels=n_kernels,
                                                          histo_figures_path=args.histo_figures_path,
                                                          threshold_neutral=threshold_neutral,
                                                          threshold_relevant=threshold_relevant,
                                                          plot_and_save_histo=plot_and_save_histo,
                                                          preliminary_clustering_dump_path=args.models_dump_path)
    regularization_test_parameters = np.arange(10, 1010, 10)
    gamma_test_parameters = np.arange(0.1, 1.1, 0.1)
    max_rate = optimal_regularization_parameter = optimal_gamma_parameter = 0
    classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=args.seq_df_path,
                                 num_test_videos=args.num_test_videos,
                                 preliminary_clustering=preliminary_clustering)
    rating_parameters_path = None
    for regularization in regularization_test_parameters:
        for gamma in gamma_test_parameters:
            print("-- Train model classifier for #kernels= " + str(n_kernels)+' with C= '+str(regularization)+
                  ' - gamma= '+str(gamma))
            classifier.train_model(percent_training_set=args.percent_training_set, regularization_parameter=regularization,
                                   gamma_parameter=gamma)
            if print_score_result == True:
                rating_parameters_path = args.rating_parameters_path + '/' + str(regularization) + '_' + str(gamma) + '.csv'
            current_rate = classifier.calculate_rate_model(percent_data_set=1-args.percent_training_set,
                                                           path_scores_parameters=rating_parameters_path)
            if current_rate > max_rate:
                max_rate = current_rate
                optimal_regularization_parameter = regularization
                optimal_gamma_parameter = gamma
    return max_rate, optimal_regularization_parameter, optimal_gamma_parameter

def clean_experiment_files(models_dump_path):
    experiment_files = glob.glob(models_dump_path + "/*.pickle")
    for f in experiment_files:
        os.remove(f)

def compare_performance_different_number_clusters(args):
    n_kernels_to_test=np.arange(100,1050,50)
    out_of_scores_path=args.scores_result_kernels_path+'/scores_number_kernels.csv'
    out_df_scores = pd.DataFrame(columns=['n_kernels', 'optimal_regularization', 'optimal_gamma', 'max_rate'])
    max_rate = optimal_n_kernels = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for n_kernels in n_kernels_to_test:
        rate, regularization_parameter, gamma_parameter = test_best_classifier_parameters(args=args, n_kernels=n_kernels,
                                                                                          print_score_result=False)
        data = np.hstack((np.array([n_kernels, regularization_parameter, gamma_parameter, rate]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
        out_df_scores.to_csv(out_of_scores_path, index=False, header=True)
        if rate > max_rate:
            max_rate = rate
            optimal_n_kernels = n_kernels
            optimal_regularization_parameter = regularization_parameter
            optimal_gamma_parameter = gamma_parameter
        clean_experiment_files(args.models_dump_path)
    return optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter

if __name__ == '__main__':
    args = get_args()
    type_tests=args.type_test
    assert type_tests == 0 or type_tests == 1
    if type_tests == 0:
        check_existing_dump_paths(files_paths=[args.coord_df_path, args.seq_df_path], dir_paths=[args.histo_figures_path,
                                               args.models_dump_path, args.rating_parameters_path])
        max_rate, optimal_regularization_parameter, optimal_gamma_parameter = test_best_classifier_parameters(args=args,
                                                                        n_kernels=args.n_kernels_preliminary_clustering,
                                                                        plot_and_save_histo=args.save_histo_figures)
        print("End test with n_kernels= " + str(args.n_kernels_preliminary_clustering) + " -- Max rate= " + str(max_rate) +
              " - Optimal_C= " + str(optimal_regularization_parameter) + " and Optimal_gamma= " + str(
            optimal_gamma_parameter))
    else:
        check_existing_dump_paths(files_paths=[args.coord_df_path, args.seq_df_path], dir_paths=[args.models_dump_path,
                                                                                        args.scores_result_kernels_path])
        optimal_n_kernels, optimal_regularization_parameter, optimal_gamma_parameter = \
            compare_performance_different_number_clusters(args)
        print("End tests - Max rate with #kernels: " + str(args.optimal_n_kernels) +
            " and classifier with C= " + str(optimal_regularization_parameter) + " and gamma= "
              + str(optimal_gamma_parameter))