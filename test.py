import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PreliminaryClustering import PreliminaryClustering
from ModelSVR import ModelSVR
from configuration import config
from utils import get_training_and_test_idx, check_existing_paths, plot_matrix, save_data_on_csv, read_dict_from_csv, plot_graph

"""Script that allows you to train an SVR using a given number of kernels for preliminary 
clustering."""

# Type of test info
fit_by_bic = config.fit_by_bic

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
if fit_by_bic:
    assert isinstance(n_kernels_GMM, list) and min(n_kernels_GMM) > 0 \
           and min(thresholds_neutral_to_test) > 0 and max(thresholds_neutral_to_test) < 1
    sub_directory = "fit_by_bic/"
else:
    assert isinstance(n_kernels_GMM, int) and n_kernels_GMM > 0 \
           and min(thresholds_neutral_to_test) > 0 and max(thresholds_neutral_to_test) < 1
    sub_directory = str(n_kernels_GMM) + "_kernels/"
# Model classifier info and paths
path_result = "data/test/" + sub_directory
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
    return model_svr.evaluate_performance()

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the kernel number of the preliminary clustering gmm.
Save the results in a csv file containing the comparison of the best scores found for each threshold """

def compare_performance_with_different_thresholds():
    out_df_scores = pd.DataFrame(columns=['Thresholds Neutral Configurations', '#clusters', 'Mean Absolute Error'])
    n_test_for_threshold = len(train_video_idx)
    thresholds_results = read_dict_from_csv(path_result_thresholds, out_df_scores, ['relevant_config', 'error'])

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
                                                               verbose=False,
                                                               threshold_neutral=threshold,
                                                               fit_by_bic=fit_by_bic)
                preliminary_clustering.execute_preliminary_clustering()
                if len(preliminary_clustering.index_relevant_configurations) > 0:
                    current_error, current_cm = generate_and_test_model(
                        threshold_neutral_configurations=threshold, preliminary_clustering=preliminary_clustering,
                        train_videos=train_videos, test_videos=test_videos)
                    threshold_sum_relevant_config += len(preliminary_clustering.index_relevant_configurations)
                    errors.append(current_error)
                    confusion_matrix += current_cm
            if len(errors) == 0:
                threshold_sum_error = "None"
            else:
                threshold_sum_error = round(sum(errors) / len(errors), 3)
                threshold_sum_relevant_config = round(threshold_sum_relevant_config / n_test_for_threshold, 2)
                thresholds_results[threshold] = {}
                thresholds_results[threshold]["error"] = threshold_sum_error
                thresholds_results[threshold]["relevant_config"] = threshold_sum_relevant_config
                
            out_df_scores = save_data_on_csv([threshold, threshold_sum_relevant_config, threshold_sum_error],
                                             out_df_scores, path_result_thresholds)
            path_current_cm = path_cm + "confusion_matrix_"+str(threshold)+".png"
            plot_matrix(cm=confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_current_cm)

    plot_graph(x=[threshold for threshold in thresholds_results.keys()],
               y=[thresholds_results[result]["error"] for result in thresholds_results],
               x_label="Threshold", y_label= "Mean Absolute Error",
               title="Mean Absolute Errors with "+str(n_kernels_GMM)+" kernels",
               file_path=path_result + "errors_graph.png")

    plot_graph(x=[threshold for threshold in thresholds_results.keys()],
               y=[thresholds_results[result]["relevant_config"] for result in thresholds_results],
               x_label="Threshold", y_label="Number of relevant configurations",
               title="Number of relevant configurations with "+str(n_kernels_GMM)+" kernels",
               file_path=path_result + "relevant_config_graph.png")

def update_results_by_BIC(dict_results_kernels):
    for n_kernels in dict_results_kernels:
        thresholds = []
        mean_absolute_errors = []
        num_relevant_config = []
        for threshold in dict_results_kernels[n_kernels]:
            thresholds.append(threshold)
            num_occ = dict_results_kernels[n_kernels][threshold]["count"]
            num_relevant_config.append(round(dict_results_kernels[n_kernels][threshold]["num_config"] / num_occ, 2))
            mean_absolute_errors.append(round(dict_results_kernels[n_kernels][threshold]["error"] / num_occ, 3))
        out_df_scores = pd.DataFrame(columns=['Thresholds Neutral Configurations', '#clusters', 'Mean Absolute Error'])
        check_existing_paths(dir_paths=[path_result + str(n_kernels)+"_kernels"])
        path_results_scores = path_result + str(n_kernels)+"_kernels/results.csv"
        for threshold_idx in np.arange(len(thresholds)):
            data = np.hstack((np.array([thresholds[threshold_idx], num_relevant_config[threshold_idx], mean_absolute_errors[threshold_idx]]).reshape(1, -1)))
            out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
        out_df_scores.to_csv(path_results_scores, index=False, header=True)

def plot_results_graph_by_BIC(dict_results_kernels):
    for n_kernels in dict_results_kernels:
        path_results_graph = path_result + str(n_kernels) + "_kernels/results_graph.png"
        plot_graph(x=[threshold for threshold in dict_results_kernels[n_kernels]],
                   y=[dict_results_kernels[n_kernels][threshold]["error"] for threshold in dict_results_kernels[n_kernels]],
                   x_label="Threshold", y_label="Mean Absolute Error",
                   title="Graphics Mean Absolute Errors",
                   file_path=path_results_graph)

def compare_performance_different_thresholds_by_BIC():
    dict_results_kernels = {}
    n_test_for_threshold = len(train_video_idx)
    for threshold_idx in np.arange(len(thresholds_neutral_to_test)):
        threshold = round(thresholds_neutral_to_test[threshold_idx], 3)
        print("Execute experiments using threshold=" + str(threshold) + "...")
        for round_idx in np.arange(n_test_for_threshold):
            print("---- Round " + str(round_idx + 1) + "/" + str(n_test_for_threshold) + "... ----")
            test_videos = test_video_idx[round_idx]
            train_videos = train_video_idx[round_idx]
            preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                           seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                           selected_lndks_idx=selected_lndks_idx,
                                                           train_video_idx=train_videos,
                                                           n_kernels=n_kernels_GMM,
                                                           covariance_type=covariance_type,
                                                           verbose=False,
                                                           threshold_neutral=threshold,
                                                           fit_by_bic=config.fit_by_bic)
            preliminary_clustering.execute_preliminary_clustering()
            n_kernels_current_GMM = preliminary_clustering.n_kernels
            num_relevant_config = len(preliminary_clustering.index_relevant_configurations)
            if n_kernels_current_GMM not in dict_results_kernels:
                dict_results_kernels[n_kernels_current_GMM] = {}
            if threshold not in dict_results_kernels[n_kernels_current_GMM]:
                dict_results_kernels[n_kernels_current_GMM][threshold] = {"error": 0, "num_config": 0, "count": 0}
            if num_relevant_config > 0:
                current_error, _ = generate_and_test_model(
                    threshold_neutral_configurations=threshold, preliminary_clustering=preliminary_clustering,
                    train_videos=train_videos, test_videos=test_videos)
                dict_results_kernels[n_kernels_current_GMM][threshold]["error"] += current_error
            dict_results_kernels[n_kernels_current_GMM][threshold]["num_config"] += num_relevant_config
            dict_results_kernels[n_kernels_current_GMM][threshold]["count"] += 1
        update_results_by_BIC(dict_results_kernels)
    plot_results_graph_by_BIC(dict_results_kernels)

if __name__ == '__main__':
    dir_paths = [path_result]
    if not fit_by_bic:
        dir_paths.append(path_cm)
    file_paths = [coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    if fit_by_bic:
        print("Execute tests with different thresholds for the neutral configurations (using GMM fitted by BIC with " +
              str(n_kernels_GMM)+" kernels, "+
            covariance_type+" covariance and "+cross_val_protocol+")")
        compare_performance_different_thresholds_by_BIC()
        print("End test with GMM fitted by BIC with "+ str(n_kernels_GMM) +" kernels: results saved in csv files with path '"+path_result+"'")
    else:
        print("Execute tests with different thresholds for the neutral configurations (using "+str(n_kernels_GMM)+" kernels, "+
            covariance_type+" covariance and "+cross_val_protocol+")")
        compare_performance_with_different_thresholds()
        print("End test with n_kernels= " + str(n_kernels_GMM) + ": results saved in a csv file with path '"+path_result+"'")
