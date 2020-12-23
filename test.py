import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PreliminaryClustering import PreliminaryClustering
from ModelSVR import ModelSVR
from configuration import config
from utils import get_training_and_test_idx, check_existing_paths, plotMatrix
import csv
import os

"""Script that allows you to train an SVR using a given number of kernels for preliminary 
clustering."""

# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
num_lndks = 66
weighted_samples = config.weighted_samples
# Features info
selected_lndks_idx = config.selected_lndks_idx
num_videos = 200
cross_val_protocol = config.cross_val_protocol
train_video_idx, test_video_idx = get_training_and_test_idx(num_videos, cross_val_protocol, seq_df_path)
# Preliminary clustering info and paths
n_kernels_GMM = config.n_kernels_GMM
covariance_type = config.covariance_type
thresholds_neutral_to_test = config.thresholds_neutral_to_test
# Model classifier info and paths
path_result = "data/test/" + str(n_kernels_GMM)+"_kernels/"
path_cm = path_result + "confusion_matrices/"
path_result_thresholds = path_result + "scores_thresholds.csv"
n_jobs = config.n_jobs

"""The procedure is performed which involves performing preliminary clustering and subsequent generation 
of SVR given the number of kernels of the GMM and the threshold for the neutral configurations
to use in the preliminary clustering"""


def generate_and_test_model(threshold_neutral_configurations,
                            preliminary_clustering, train_videos, test_videos):
    assert 0 < threshold_neutral_configurations < 1
    model_svr = ModelSVR(seq_df_path=seq_df_path,
                         train_video_idx=train_videos, test_video_idx=test_videos,
                         preliminary_clustering=preliminary_clustering, weighted_samples=weighted_samples,
                         verbose=False)
    model_svr.train_SVR(train_by_max_score=True, n_jobs=n_jobs)
    return model_svr.calculate_rate_model()

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the kernel number of the preliminary clustering gmm.
Save the results in a csv file containing the comparison of the best scores found for each threshold """

def compare_performance_different_thresholds():
    out_df_scores = pd.DataFrame(columns=['Thresholds Neutral Configurations', '#clusters', 'Mean Absolute Error'])
    n_test_for_threshold = len(train_video_idx)
    thresholds_results = {}
    if os.path.isfile(path_result_thresholds):
        with open(path_result_thresholds, 'r') as thresholds_rslt_file:
            reader = csv.reader(thresholds_rslt_file)
            for idx, row in enumerate(reader):
                if idx > 0:
                    thresholds_results[float(row[0])] = {}
                    thresholds_results[float(row[0])]['relevant_config'] = float(row[1])
                    thresholds_results[float(row[0])]['error'] = float(row[2])
                    data = np.hstack((np.array([row[0], row[1], row[2]]).reshape(1, -1)))
                    out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
    for threshold_idx in np.arange(0, len(thresholds_neutral_to_test)):
        threshold = round(thresholds_neutral_to_test[threshold_idx], 3)
        if threshold not in thresholds_results:
            errors = []
            threshold_sum_relevant_config = 0
            confusion_matrix = np.zeros(shape=(11, 11))
            print("Execute experiments using threshold=" + str(threshold) + "...")
            for test_idx in np.arange(0, n_test_for_threshold):
                print("---- Round "+str(test_idx+1)+"/"+str(n_test_for_threshold)+"... ----")
                test_videos = test_video_idx[test_idx]
                train_videos = train_video_idx[test_idx]
                preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                               seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                               selected_lndks_idx=selected_lndks_idx,
                                                               train_video_idx=train_videos,
                                                               n_kernels=n_kernels_GMM,
                                                               covariance_type=covariance_type,
                                                               verbose=False)
                preliminary_clustering.execute_preliminary_clustering(threshold_neutral=threshold)
                if len(preliminary_clustering.index_relevant_configurations) > 0:
                    current_error, current_cm = generate_and_test_model(
                        threshold_neutral_configurations=threshold, preliminary_clustering=preliminary_clustering,
                        train_videos=train_videos, test_videos=test_videos)
                    errors.append(current_error)
                    threshold_sum_relevant_config += len(preliminary_clustering.index_relevant_configurations)
                    confusion_matrix += current_cm
            if len(errors) == 0:
                threshold_sum_error = "None"
            else:
                threshold_sum_error = round(sum(errors) / len(errors), 3)
                threshold_sum_relevant_config = int(threshold_sum_relevant_config / n_test_for_threshold)
                thresholds_results[threshold] = {}
                thresholds_results[threshold]["error"] = threshold_sum_error
                thresholds_results[threshold]["relevant_config"] = threshold_sum_relevant_config
            data = np.hstack((np.array([threshold, threshold_sum_relevant_config, threshold_sum_error]).reshape(1, -1)))
            out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
            out_df_scores.to_csv(path_result_thresholds, index=False, header=True)
            path_current_cm = path_cm + "confusion_matrix_"+str(threshold)+".png"
            plotMatrix(cm=confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_current_cm)
    path_errors_graph = path_result + "errors_graph.png"
    plt.plot(thresholds_neutral_to_test, [thresholds_results[result]["error"] for result in thresholds_results], color="blue")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Threshold")
    plt.title("Graphics Mean Absolute Errors")
    plt.savefig(path_errors_graph)
    plt.close()

if __name__ == '__main__':
    assert n_kernels_GMM > 0
    dir_paths = [path_result, path_cm]
    file_paths = [coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    print("Execute tests with different thresholds for the neutral configurations (using "+str(n_kernels_GMM)+" kernels, "+
    covariance_type+" covariance and "+cross_val_protocol+")")
    compare_performance_different_thresholds()
    print("End test with n_kernels= " + str(n_kernels_GMM) + ": results saved in a csv file with path '"+path_result+"'")
