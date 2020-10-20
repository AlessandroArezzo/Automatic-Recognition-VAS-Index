import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
from configuration import config_tests
from utils import get_training_and_test_idx, check_existing_paths

# Dataset info
coord_df_path = "data/dataset/2d_skeletal_data_unbc_coords.csv"
seq_df_path = "data/dataset/2d_skeletal_data_unbc_sequence.csv"
num_lndks = 66
# Features info
selected_lndks_idx = config_tests.selected_lndks_idx
num_videos = 200
cross_val_protocol = config_tests.cross_val_protocol
train_video_idx, test_video_idx = get_training_and_test_idx(num_videos, cross_val_protocol)
# Preliminary clustering info and paths
n_kernels_GMM = config_tests.n_kernels_GMM
thresholds_neutral_to_test = config_tests.thresholds_neutral_to_test
# Model classifier info and paths
type_classifier = config_tests.type_classifier
path_result = "data/test/" + str(n_kernels_GMM)+"_kernels/"
path_result_thresholds = path_result + "scores_thresholds_"+type_classifier+".csv"
path_result_graph = path_result + "scores_graphics_"+type_classifier+".png"

"""The procedure is performed which involves performing preliminary clustering and subsequent generation 
of the classifier (SVM or SVR) given the number of kernels of the GMM and the threshold for the neutral configurations
to use in the preliminary clustering"""


def generate_and_test_model(threshold_neutral_configurations,
                            preliminary_clustering, train_videos, test_videos):
    assert 0 < threshold_neutral_configurations < 1
    model_classifier = ModelClassifier(type_classifier=type_classifier, seq_df_path=seq_df_path,
                                 train_video_idx=train_videos, test_video_idx=test_videos,
                                preliminary_clustering=preliminary_clustering, verbose=False)
    model_classifier.train_model(train_by_max_score=True)
    score = model_classifier.calculate_rate_model()
    return score

"""Compare the best scores obtained by varying the thresholds used for the neutral configurations in the 
preliminary clustering. 
The respective value of the parameter input to the script is used as the kernel number of the preliminary clustering gmm.
Save the results in a csv file containing the comparison of the best scores found for each threshold """

def compare_performance_different_thresholds():
    out_df_scores = pd.DataFrame(columns=['Thresholds Neutral Configurations', 'Mean Absolute Error'])
    min_error = 100000
    optimal_thresholds = 0
    #n_test_for_threshold = len(train_video_idx)
    n_test_for_threshold = 5
    mean_errors = []
    for threshold_idx in np.arange(0,len(thresholds_neutral_to_test)):
        threshold = thresholds_neutral_to_test[threshold_idx]
        threshold = round(threshold, 3)
        threshold_sum_error = 0
        print("Execute experiments using threshold=" + str(threshold) + "...")
        for test_idx in np.arange(0, n_test_for_threshold):
            print("---- Test "+str(test_idx+1)+"/"+str(n_test_for_threshold)+"... ----")
            test_videos = test_video_idx[test_idx]
            train_videos = train_video_idx[test_idx]
            preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                           seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                           selected_lndks_idx=selected_lndks_idx,
                                                           train_video_idx=train_videos,
                                                           n_kernels=n_kernels_GMM, verbose=False)
            preliminary_clustering.execute_preliminary_clustering(threshold_neutral=threshold)
            if len(preliminary_clustering.index_relevant_configurations) == 0:
                threshold_error = "None"
            else:
                threshold_error = generate_and_test_model(
                    threshold_neutral_configurations=threshold, preliminary_clustering=preliminary_clustering,
                    train_videos=train_videos, test_videos=test_videos)
            threshold_sum_error += threshold_error
        threshold_sum_error /= n_test_for_threshold
        threshold_sum_error = round(threshold_sum_error, 3)
        mean_errors.append(threshold_sum_error)
        data = np.hstack((np.array([threshold, threshold_sum_error]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),ignore_index=True)
        out_df_scores.to_csv(path_result_thresholds, index=False, header=True)
        if threshold_sum_error < min_error:
            min_error = threshold_sum_error
            optimal_thresholds = threshold
    plt.plot(thresholds_neutral_to_test, mean_errors, color="blue")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Threshold")
    plt.title("Graphics Mean Absolute Errors")
    plt.savefig(path_result_graph)
    plt.close()
    return optimal_thresholds, min_error

if __name__ == '__main__':
    assert n_kernels_GMM > 0 and (type_classifier == 'SVM' or type_classifier == 'SVR')
    dir_paths = [path_result]
    file_paths = [coord_df_path, seq_df_path]
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    print("Execute tests with different thresholds for the neutral configurations (using "+str(n_kernels_GMM)+" kernels "
    "and "+cross_val_protocol+")")
    max_threshold, min_error = compare_performance_different_thresholds()
    print("End test with n_kernels= " + str(n_kernels_GMM) + ": best threshold= " + str(max_threshold)+ " with Mean "
                                                                            "Absolute Error = "+str(min_error))
