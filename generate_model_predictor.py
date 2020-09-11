import argparse
import os
import numpy as np
from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier

"""Script that allows you to train a classifier (SVM or SVR) using a given number of kernels for preliminary 
clustering, gamma and regularization parameters of the model to be fitted. The model is saved in a pickle file."""


def get_args():
    parser = argparse.ArgumentParser()
    # Clustering info
    parser.add_argument('-load_preliminary_clustering', "--load_preliminary_clustering",
                        help="Determine if preliminary clustering is must be readed from a pickle file", default=True,
                        type=bool)
    parser.add_argument('-dump_preliminary_clustering', "--dump_preliminary_clustering",
                        help="Determine if preliminary clustering is must be saved in a pickle file", default=True,
                        type=bool)
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=200, type=int)
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
    parser.add_argument('-regularization_parameter', "--regularization_parameter",
                        help="Regularization_parameter",
                        default=1, type=int)
    parser.add_argument('-gamma_parameter', "--gamma_parameter",
                        help="Gamma_parameter",
                        default='scale')
    return parser.parse_args()


args = get_args()
# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
num_lndks = 66
percent_training_set = 0.85
# Features info
selected_lndks_idx = np.arange(66)
num_test_videos = 200
# Preliminary clustering info and paths
n_kernels_preliminary_clustering = args.n_kernels_preliminary_clustering
threshold_neutral = args.threshold_neutral
threshold_relevant = args.threshold_relevant
dump_preliminary_clustering = args.dump_preliminary_clustering
load_preliminary_clustering = args.load_preliminary_clustering
save_histo_figures = args.save_histo_figures
sub_directory = str(n_kernels_preliminary_clustering) + "_kernels"
histo_figures_path = "data/classifier/" + sub_directory + "/figures/histograms"  # For save_histo_figures = True
preliminary_clustering_path = "data/classifier/" + sub_directory + "/preliminary_clustering.pickle"
# Model classifier info and paths
type_classifier = args.type_classifier
regularization_parameter = args.regularization_parameter
gamma_parameter = args.gamma_parameter
classifier_path = "data/classifier/" + sub_directory + "/" + type_classifier + "_classifier.pickle"
path_scores_parameters = "data/classifier/" + sub_directory + "/" + str(n_kernels_preliminary_clustering) + "_" \
                         + str(args.regularization_parameter) + "_" + str(args.gamma_parameter) + "_scores.csv"

if __name__ == '__main__':
    args = get_args()
    n_kernels_preliminary_clustering = args.n_kernels_preliminary_clustering
    if args.load_preliminary_clustering:
        print("Read preliminary clustering from file...")
        preliminary_clustering = PreliminaryClustering.load_from_pickle(pickle_path=preliminary_clustering_path)
    else:
        print("Execute preliminary clustering with #kernels=" + str(n_kernels_preliminary_clustering) + "...")
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       num_test_videos=num_test_videos,
                                                       n_kernels=n_kernels_preliminary_clustering,
                                                       threshold_neutral=threshold_neutral,
                                                       threshold_relevant=threshold_relevant)
        preliminary_clustering_dump_path = preliminary_clustering_path if dump_preliminary_clustering == True else None
        preliminary_clustering.execute_preliminary_clustering(
            preliminary_clustering_dump_path=preliminary_clustering_dump_path)
    classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 num_test_videos=num_test_videos,
                                 preliminary_clustering=preliminary_clustering)
    print("Train and save " + type_classifier + " model...")
    classifier.train_model(percent_training_set=percent_training_set, regularization_parameter=regularization_parameter,
                           gamma_parameter=gamma_parameter, classifier_dump_path=classifier_path)
    print(args.type_classifier + " trained and saved in model_classifier_path")
    print("Calculate scores for trained classifier...")
    rate = classifier.calculate_rate_model(percent_data_set=1 - percent_training_set,
                                           path_scores_parameters=path_scores_parameters)
    print("Rate classifier is " + str(rate))
