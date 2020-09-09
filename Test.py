from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    # Dataset info
    parser.add_argument('-coord_df_path' , "--coord_df_path", help="Path coordinates df" ,
                        default='data/dataset/2d_skeletal_data_unbc_coords.csv')
    parser.add_argument("-seq_df_path" , "--seq_df_path" , help="Path sequences df",
                        default = 'data/dataset/2d_skeletal_data_unbc_sequence.csv')
    parser.add_argument('-num_lndks' , "--num_lndks", help="Number of facial landmarks" , default=66, type=int)
    parser.add_argument('-percent_training_set', "--percent_training_set", help="Percent of data to use for training set",
                        default=0.80, type=float)
    #Features info
    parser.add_argument('-selected_lndks_idx', "--selected_lndks_idx", help="Number of facial landmarks",
                        default=np.arange(66), type=int)
    parser.add_argument('-num_test_videos' , "--num_test_videos", help="Number of videos to use for test" ,
                        default=200, type=int)
    #Clustering info
    parser.add_argument('-threshold_neutral', "--threshold_neutral", help="Threshold for neutral configuration in clustering",
                        default=0.3, type=float)
    parser.add_argument('-threshold_relevant', "--threshold_relevant", help="Threshold for relevant configuration in clustering",
                        default=0.2, type=float)
    parser.add_argument('-save_histo_figures' , "--save_histo_figures",
                        help="Determines if histograms are to be saved during preliminary clustering phases" ,
                        default=False, type=bool)
    # Path files generated during preliminary clustering phase
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=150, type=int)
    n_kernels = parser.parse_args().n_kernels_preliminary_clustering
    sub_directory=str(n_kernels)+"_kernels"
    parser.add_argument('-histo_figures_path' , "--histo_figures_path", help="Path histograms figures" ,
                        default='data/test/'+sub_directory+'/figures/histograms')
    parser.add_argument('-preliminary_clustering_path', "--preliminary_clustering_path",
                        help="Path file preliminary clustering",
                        default='data/test/' + sub_directory + '/preliminary_clustering.pickle')
    parser.add_argument('-classifier_model_path', "--classifier_model_path",
                        help="Path file classifier model",
                        default='data/test/' + sub_directory + '/classifier_model.pickle')
    parser.add_argument('-rating_parameters_path', "--rating_parameters_path",
                        help="Path rating classifiers parameters",
                        default='data/test/' + sub_directory + '/test_SVM_parameters/rate_classifiers/')
    return parser.parse_args()


def test_best_classifier_parameters(args):
    n_kernels=args.n_kernels_preliminary_clustering
    print("Experiments for #kernels= " + str(n_kernels))
    print("-- Preliminary clustering for #kernels= " + str(n_kernels))
    preliminary_clustering = PreliminaryClustering()
    preliminary_clustering.execute_preliminary_clustering(coord_df_path=args.coord_df_path,
                                                   seq_df_path=args.seq_df_path, num_lndks=args.num_lndks,
                                                   selected_lndks_idx=args.selected_lndks_idx,
                                                   num_test_videos=args.num_test_videos,
                                                   n_kernels=args.n_kernels_preliminary_clustering,
                                                   histo_figures_path=args.histo_figures_path,
                                                   threshold_neutral=args.threshold_neutral,
                                                   threshold_relevant=args.threshold_relevant,
                                                   plot_and_save_histo=args.save_histo_figures,
                                                   preliminary_clustering_dump_path=args.preliminary_clustering_path)
    regularization_test_parameters = np.arange(10, 1010, 10)
    gamma_test_parameters = np.arange(0.1, 1.1, 0.1)
    max_rate = optimal_regularization_parameter = optimal_gamma_parameter = 0
    for regularization in regularization_test_parameters:
        for gamma in gamma_test_parameters:
            print("-- Clustering sequences for #kernels= " + str(n_kernels))
            classifier = ModelClassifier(type_classifier="SVM", regularization_parameter=regularization,
                                         gamma_parameter=gamma)
            classifier.train_model(seq_df_path=args.seq_df_path, num_test_videos=args.num_test_videos,
                                   preliminary_clustering=preliminary_clustering,
                                   percent_training_set=args.percent_training_set)
            rating_parameters_path=args.rating_parameters_path+'/'+str(regularization)+'_'+str(gamma)+'.pickle'
            current_rate = classifier.calculate_rate_model(path_scores_parameters=rating_parameters_path)
            if current_rate > max_rate:
                max_rate = current_rate
                optimal_regularization_parameter = regularization
                optimal_gamma_parameter = gamma
    return max_rate, optimal_regularization_parameter, optimal_gamma_parameter

if __name__ == '__main__':
    args = get_args()
    max_rate, optimal_regularization_parameter, optimal_gamma_parameter = test_best_classifier_parameters(args)
    print("End test with n_kernels= " + str(args.n_kernels_test) + " -- Max rate= " + str(max_rate) +
          " - Optimal_C= " + str(optimal_regularization_parameter) + " - Optimal_gamma= " + str(
        optimal_gamma_parameter))