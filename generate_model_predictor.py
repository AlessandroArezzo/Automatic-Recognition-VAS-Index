import numpy as np
from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
from configuration import config_generate_model
import os
"""Script that allows you to train a classifier (SVM or SVR) using a given number of kernels for preliminary 
clustering, gamma and regularization parameters of the model to be fitted. The model is saved in a pickle file."""

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
selected_lndks_idx = config_generate_model.selected_lndks_idx
num_test_videos = 200
train_video_idx = np.arange(0,num_test_videos)[:int(percent_training_set * num_test_videos)]
test_video_idx = np.arange(0,num_test_videos)[int(percent_training_set * num_test_videos):num_test_videos]
# Preliminary clustering info and paths
n_kernels_GMM = config_generate_model.n_kernels_GMM
threshold_neutral = config_generate_model.threshold_neutral
dump_preliminary_clustering = config_generate_model.dump_preliminary_clustering
load_preliminary_clustering = config_generate_model.load_preliminary_clustering
save_histo_figures = config_generate_model.save_histo_figures
sub_directory = str(n_kernels_GMM) + "_kernels"
histo_figures_path = "data/classifier/" + sub_directory + "/figures/histograms"
preliminary_clustering_path = "data/classifier/" + sub_directory + "/preliminary_clustering.pickle"
# Model classifier info and paths
type_classifier = config_generate_model.type_classifier
train_by_max_score = config_generate_model.train_by_max_score
regularization_parameter = config_generate_model.regularization_parameter
gamma_parameter = config_generate_model.gamma_parameter
classifier_path = "data/classifier/" + sub_directory + "/" + type_classifier + "_classifier.pickle"
path_scores_parameters = "data/classifier/" + sub_directory + "/" + type_classifier + "_scores.csv"

if __name__ == '__main__':
    assert n_kernels_GMM > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR') \
           and 0 < threshold_neutral < 1
    dir_paths = ['data/classifier/'+sub_directory+'/']
    if save_histo_figures:
        dir_paths.append('data/classifier/'+sub_directory+'//figures/histograms/')
    file_paths=[coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    if load_preliminary_clustering:
        print("Read preliminary clustering from file...")
        preliminary_clustering = PreliminaryClustering.load_from_pickle(pickle_path=preliminary_clustering_path)
    else:
        print("Execute preliminary clustering with #kernels=" + str(n_kernels_GMM) + "...")
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       train_video_idx=train_video_idx,
                                                       n_kernels=n_kernels_GMM)
        preliminary_clustering_dump_path = preliminary_clustering_path if dump_preliminary_clustering == True else None
        preliminary_clustering.execute_preliminary_clustering(plot_and_save_histo=save_histo_figures,
           histo_figures_path=histo_figures_path, preliminary_clustering_dump_path=preliminary_clustering_dump_path,
           threshold_neutral=threshold_neutral)
    classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 train_video_idx=train_video_idx,
                                 test_video_idx=test_video_idx,
                                 preliminary_clustering=preliminary_clustering)
    print("Train and save " + type_classifier + " model...")
    classifier.train_model(regularization_parameter=regularization_parameter,
                           gamma_parameter=gamma_parameter, train_by_max_score=train_by_max_score,
                           classifier_dump_path=classifier_path)
    print(type_classifier + " trained and saved in model_classifier_path")
    print("Calculate scores for trained classifier...")
    rate = classifier.calculate_rate_model(path_scores_parameters=path_scores_parameters)
    print("Rate classifier is " + str(rate))
