import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PreliminaryClustering import PreliminaryClustering
from ModelSVR import ModelSVR
from configuration import config_generate_model
from utils import get_training_and_test_idx, check_existing_paths
"""Script that allows you to train an SVR using a given number of kernels for preliminary 
clustering."""


# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
num_lndks = 66
# Features info
selected_lndks_idx = config_generate_model.selected_lndks_idx
num_videos = 200
cross_val_protocol = config_generate_model.cross_val_protocol
train_video_idx, test_video_idx = get_training_and_test_idx(num_videos, cross_val_protocol)
# Preliminary clustering info and paths
n_kernels_GMM = config_generate_model.n_kernels_GMM
threshold_neutral = config_generate_model.threshold_neutral
save_histo_figures = config_generate_model.save_histo_figures
sub_directory = str(n_kernels_GMM) + "_kernels"
path_histo_figures = "data/classifier/" + sub_directory + "/histo_figures/"
preliminary_clustering_path = "data/classifier/" + sub_directory + "/preliminary_clustering.pickle"
# Model classifier info and paths
path_results = "data/classifier/" + sub_directory + "/"
path_errors = path_results + "/errors_tests/"

if __name__ == '__main__':
    assert n_kernels_GMM > 0 and (threshold_neutral == None or 0 < threshold_neutral < 1)
    dir_paths = [path_errors]
    if save_histo_figures:
        dir_paths.append(path_histo_figures)
    file_paths = [coord_df_path, seq_df_path]
    out_df_scores = pd.DataFrame(columns=['Num_test', 'Mean Absolute Error', 'Accuracy(%)'])
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    n_test = len(train_video_idx)
    errors = []
    accuracy = []
    if threshold_neutral == None:
        print("Generate and test models with " + str(n_kernels_GMM) + " kernels GMM, default threshold and using + cross_val_protocol")
    else:
        print("Generate and test models with "+str(n_kernels_GMM)+" kernels GMM, threshold = "+str(threshold_neutral)+ " and using "+cross_val_protocol )
    for test_idx in np.arange(0, n_test):
        print("- Test "+str(test_idx+1)+"/"+str(n_test)+" -")
        test_videos = test_video_idx[test_idx]
        train_videos = train_video_idx[test_idx]
        print("-- Execute preliminary clustering using " + str(n_kernels_GMM) + " kernels GMM... --")
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       train_video_idx=train_videos,
                                                       n_kernels=n_kernels_GMM)
        path_histo_current = path_histo_figures + "test_"+str(test_idx)+"_"
        preliminary_clustering.execute_preliminary_clustering(plot_and_save_histo=save_histo_figures,
           histo_figures_path=path_histo_current, threshold_neutral=threshold_neutral)
        if len(preliminary_clustering.index_relevant_configurations) == 0:
            print("-- No relevant configurations were found using "+str(n_kernels_GMM)+" kernels and "+str(threshold_neutral)+" for the threshold of neutral configurations "
                  "(try to lower the threshold by analyzing the histograms produced by clustering in the test module )--")
            current_error = current_accuracy = "None"
        else:
            model_svr = ModelSVR(seq_df_path=seq_df_path,
                                 train_video_idx=train_videos,
                                 test_video_idx=test_videos,
                                 preliminary_clustering=preliminary_clustering)
            print("-- Train and save SVR model... --")
            model_svr.train_SVR(train_by_max_score=True)
            print("-- Calculate scores for trained SVR... --")
            current_test_path_error = path_errors+"test_"+str(test_idx)+".csv"
            current_error, current_accuracy = model_svr.calculate_rate_model(path_scores_parameters=current_test_path_error)
            errors.append(current_error)
            accuracy.append(current_accuracy)
            print("-- Absolute Error: " + str(current_error)+" / Accuracy: " + str(current_accuracy)+"% --")
        data = np.hstack((np.array([test_idx+1, current_error, current_accuracy]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                             ignore_index=True)
    path_results_csv = path_results + "results.csv"
    path_errors_histo = path_results + "graphics_errors.png"
    path_accuracy_histo = path_results + "graphics_accuracy.png"
    out_df_scores.to_csv(path_results_csv, index=False, header=True)
    print("Results saved in a csv file on path '" + path_results_csv)
    mean_error = sum(errors)/n_test
    mean_error = round(mean_error, 3)
    plt.bar(np.arange(1, n_test+1), errors, color="blue")
    plt.axhline(y=mean_error, xmin=0, xmax=n_test+1, color="red", label='Mean Absolute Error: '+str(mean_error))
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Num test")
    plt.title("Graphics Mean Absolute Errors")
    plt.legend()
    plt.savefig(path_errors_histo)
    plt.close()
    mean_accuracy = sum(accuracy) / n_test
    mean_accuracy = round(mean_accuracy, 3)
    plt.bar(np.arange(1, n_test + 1), accuracy, color="blue")
    plt.axhline(y=mean_accuracy, xmin=0, xmax=n_test + 1, color="red", label='Accuracy: ' + str(mean_accuracy))
    plt.ylabel("Accuracy")
    plt.xlabel("Num test")
    plt.title("Graphics Accuracy Model")
    plt.legend()
    plt.savefig(path_accuracy_histo)
    plt.close()
    print("Histogram of the results generated saved in a png file on path '" + path_results)
    print("Mean Absolute Error: " + str(mean_error))
    print("Accuracy: " + str(mean_accuracy)+"%")