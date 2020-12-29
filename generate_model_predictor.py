import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

from PreliminaryClustering import PreliminaryClustering
from ModelSVR import ModelSVR
from configuration import config
from utils import get_training_and_test_idx, check_existing_paths, plotMatrix
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
threshold_neutral = config.threshold_neutral
save_histo_figures = config.save_histo_figures
sub_directory = str(n_kernels_GMM) + "_kernels"
path_histo_figures = "data/classifier/" + sub_directory + "/histo_figures/"
preliminary_clustering_path = "data/classifier/" + sub_directory + "/preliminary_clustering.pickle"
# Model classifier info and paths
path_results = "data/classifier/" + sub_directory + "/"
path_errors = path_results + "errors_tests/"
path_gmm_means = path_results + "/gmm_means/"
path_confusion_matrices = path_results + "confusion_matrices/"
n_jobs = config.n_jobs
model = manifold.MDS(n_components=2, metric=True, n_init=4, random_state=1, max_iter=200, dissimilarity='euclidean')

if __name__ == '__main__':
    assert n_kernels_GMM > 0 and (threshold_neutral == None or 0 < threshold_neutral < 1)
    dir_paths = [path_results, path_errors, path_confusion_matrices, path_gmm_means]
    if save_histo_figures:
        dir_paths.append(path_histo_figures)
    file_paths = [coord_df_path, seq_df_path]
    out_df_scores = pd.DataFrame(columns=['#round', '#num_clusters', 'Mean Absolute Error'])
    check_existing_paths(dir_paths=dir_paths, file_paths=file_paths)
    n_test = len(train_video_idx)
    path_results_csv = path_results + "results.csv"
    path_conf_matrix_csv = path_results + "confusion_matrix.csv"
    path_histo_current = None
    errors = []
    confusion_matrix = np.zeros(shape=(11, 11))
    if threshold_neutral == None:
        print("Generate and test models with " + str(n_kernels_GMM) + " kernels GMM, "+covariance_type+" covariance, default threshold and using " + cross_val_protocol)
    else:
        print("Generate and test models with "+str(n_kernels_GMM)+" kernels GMM, "+covariance_type+" covariance, threshold = "+str(threshold_neutral)+ " and using "+cross_val_protocol )
    for test_idx in np.arange(0, n_test):
        print("- Round "+str(test_idx+1)+"/"+str(n_test)+" -")
        test_videos = test_video_idx[test_idx]
        train_videos = train_video_idx[test_idx]
        print("-- Execute preliminary clustering using " + str(n_kernels_GMM) + " kernels GMM... --")
        preliminary_clustering = PreliminaryClustering(coord_df_path=coord_df_path,
                                                       seq_df_path=seq_df_path, num_lndks=num_lndks,
                                                       selected_lndks_idx=selected_lndks_idx,
                                                       train_video_idx=train_videos,
                                                       n_kernels=n_kernels_GMM,
                                                       covariance_type=covariance_type,)
        if save_histo_figures == True:
            path_histo_current = path_histo_figures + "test_"+str(test_idx)+"_"
        preliminary_clustering.execute_preliminary_clustering(histo_figures_path=path_histo_current, threshold_neutral=threshold_neutral)
        num_relevant_config = len(preliminary_clustering.index_relevant_configurations)
        if num_relevant_config == 0:
            print("-- No relevant configurations were found using "+str(n_kernels_GMM)+" kernels and "+str(threshold_neutral)+" for the threshold of neutral configurations "
                  "(try to lower the threshold by analyzing the histograms produced by clustering in the test module )--")
            current_error = current_accuracy = "None"
        else:
            print("-- Preliminary clustering ended: "+str(num_relevant_config)+" relevant clusters founded --")
            model_svr = ModelSVR(seq_df_path=seq_df_path,
                                 train_video_idx=train_videos,
                                 test_video_idx=test_videos,
                                 preliminary_clustering=preliminary_clustering,
                                 weighted_samples=weighted_samples)
            print("-- Train and save SVR model... --")
            model_svr.train_SVR(train_by_max_score=True, n_jobs=n_jobs)
            print("-- Calculate scores for trained SVR... --")
            current_test_path_error = path_errors+"errors_test_"+str(test_idx)+".csv"
            current_path_cm = path_confusion_matrices + "conf_matrix_test_" + str(test_idx) + ".png"
            current_error, current_confusion_matrix = model_svr.evaluate_performance_model(path_scores_parameters=current_test_path_error,
                                                                                     path_scores_cm=current_path_cm)
            errors.append(current_error)
            print("-- Mean Absolute Error: " + str(current_error)+" --")
            confusion_matrix += current_confusion_matrix
        data_df_scores = np.hstack((np.array([test_idx+1, num_relevant_config, current_error]).reshape(1, -1)))
        out_df_scores = out_df_scores.append(pd.Series(data_df_scores.reshape(-1), index=out_df_scores.columns),
                                             ignore_index=True)
        out_df_scores.to_csv(path_results_csv, index=False, header=True)
        gmm_means = preliminary_clustering.gmm.means
        columns = ["#kernel"] + ["ldk #"+str(ldks_idx) for ldks_idx in selected_lndks_idx]
        out_gmm_means = pd.DataFrame(columns=[columns])
        for kernel_idx in np.arange(len(gmm_means)):
            data_gmm_means = np.hstack((np.array([kernel_idx] + [center for center in gmm_means[kernel_idx]]).reshape(1, -1)))
            out_gmm_means = out_gmm_means.append(pd.Series(data_gmm_means.reshape(-1), index=out_gmm_means.columns),
                                                 ignore_index=True)
        current_path_gmm_means = path_gmm_means + "gmm_means_test_" + str(test_idx) + ".csv"
        out_gmm_means.to_csv(current_path_gmm_means, index=False, header=True)
        data_transformed = model.fit_transform(gmm_means)
        plt.plot(data_transformed[:, 0], data_transformed[:, 1], '.b')
        current_path_img_clusters = path_gmm_means + "gmm_clusters_test_" + str(test_idx) + ".png"
        for k in np.arange(data_transformed.shape[0]):
            plt.annotate(str(k), (data_transformed[k, 0], data_transformed[k, 1]))
        plt.title('Position of %d clusters remapped in 2D with MSD' % (data_transformed.shape[0]))
        plt.savefig(current_path_img_clusters)
        plt.close()
    mean_error = sum(errors) / n_test
    mean_error = round(mean_error, 3)
    print("Mean Absolute Error: " + str(mean_error))

    path_errors = path_results + "graphics_errors.png"
    path_conf_matrix = path_results + "confusion_matrix.png"
    print("Mean absolute errors detected at each round saved in a csv file on path '" + path_results_csv+"'")
    print("Confusion matrices detected at each round saved in png files on path '" + path_confusion_matrices+"'")

    plotMatrix(cm=confusion_matrix, labels=np.arange(0, 11), normalize=True, fname=path_conf_matrix)
    print("Overall confusion matrix saved in png files on path '" + path_conf_matrix+"'")
    plt.bar(np.arange(1, n_test+1), errors, color="blue")
    plt.axhline(y=mean_error, xmin=0, xmax=n_test+1, color="red", label='Mean Absolute Error: '+str(mean_error))
    plt.ylabel("Average of the Mean Absolute Error")
    plt.xlabel("Num round")
    plt.title("Mean Absolute Errors")
    plt.legend()
    plt.savefig(path_errors)
    plt.close()
    print("Histogram of the mean absolute error detected saved in a png file on path '" + path_results+"'")

