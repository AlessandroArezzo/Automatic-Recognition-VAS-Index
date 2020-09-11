import argparse
import os
import numpy as np
from PreliminaryClustering import PreliminaryClustering
from ModelClassifier import ModelClassifier

"""Script that allows you to train a classifier (SVM or SVR) using a given number of kernels for preliminary 
clustering, gamma and regularization parameters of the model to be fitted. The model is saved in a pickle file."""
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
    parser.add_argument('-load_preliminary_clustering', "--load_preliminary_clustering",
                        help="Determine if preliminary clustering is must be readed from file", default=False, type=bool)
    parser.add_argument('-n_kernels_preliminary_clustering', "--n_kernels_preliminary_clustering",
                        help="Number of kernels to use for GMM of the preliminary clustering", default=200, type=int)
    parser.add_argument('-threshold_neutral', "--threshold_neutral",
                        help="Threshold for neutral configuration in preliminary clustering",
                        default=0.3, type=float)
    parser.add_argument('-threshold_relevant', "--threshold_relevant",
                        help="Threshold for relevant configuration in preliminary clustering",
                        default=0.2, type=float)
    #Classifier info
    parser.add_argument('-type_classifier', "--type_classifier",
                        help="Determines type of classifier to use: 'SVM' or 'SVR",
                        default='SVM')
    parser.add_argument('-regularization_parameter', "--regularization_parameter",
                        help="Regularization_parameter",
                        default=1, type=int)
    parser.add_argument('-gamma_parameter', "--gamma_parameter",
                        help="Gamma_parameter",
                        default='scale')
    # Path files generated
    parser.add_argument('-models_path', "--models_path",
                        help="Path for preliminary clustering and model classifier",
                        default='data/classifier')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    assert os.path.isdir(args.models_path)
    n_kernels_preliminary_clustering=args.n_kernels_preliminary_clustering
    if args.load_preliminary_clustering:
        print("Read preliminary clustering from file...")
        preliminary_clustering = PreliminaryClustering.load_from_pickle(args.models_path+'/'+str(n_kernels_preliminary_clustering)+
                                                                                             '_kernels_preliminary_clustering.pickle')
    else:
        print("Execute preliminary clustering with #kernels="+str(n_kernels_preliminary_clustering)+"...")
        preliminary_clustering = PreliminaryClustering(coord_df_path=args.coord_df_path,
                                                          seq_df_path=args.seq_df_path, num_lndks=args.num_lndks,
                                                          selected_lndks_idx=args.selected_lndks_idx,
                                                          num_test_videos=args.num_test_videos,
                                                          n_kernels=n_kernels_preliminary_clustering,
                                                          threshold_neutral=args.threshold_neutral,
                                                          threshold_relevant=args.threshold_relevant)
        preliminary_clustering.execute_preliminary_clustering(preliminary_clustering_dump_path=args.models_path)
    model_classifier_path = args.models_path+'/'+args.type_classifier+'_'+str(n_kernels_preliminary_clustering)+'_kernels.pickle'
    classifier = ModelClassifier(type_classifier=args.type_classifier, seq_df_path=args.seq_df_path,
                                 num_test_videos=args.num_test_videos,
                                 preliminary_clustering=preliminary_clustering)
    print("Train and save "+args.type_classifier+" model...")
    classifier.train_model(percent_training_set=args.percent_training_set, regularization_parameter=args.regularization_parameter,
                           gamma_parameter=args.gamma_parameter, classifier_dump_path=model_classifier_path)
    print(args.type_classifier+" trained and saved in model_classifier_path")
    path_scores_parameters = args.models_path+'/'+str(n_kernels_preliminary_clustering)+"_"\
                             +str(args.regularization_parameter)+'_'+str(args.gamma_parameter)+'_scores.csv'
    print("Calculate scores for trained classifier...")
    rate = classifier.calculate_rate_model(percent_data_set=1-args.percent_training_set,
                                           path_scores_parameters=path_scores_parameters)
    print("Rate classifier is "+str(rate))