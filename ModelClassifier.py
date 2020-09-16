from FisherVectors import FisherVectorGMM
import pandas as pd
import pickle
import numpy as np
from sklearn import svm

"""Class that deals with training a classifier (SVM or SVR) starting from the relevant configurations 
to characterize the VAS index obtained during the preliminary clustering phase. """
class ModelClassifier:

    def __init__(self, seq_df_path, num_test_videos, preliminary_clustering, type_classifier='SVM'):
        assert type_classifier == 'SVM' or type_classifier == 'SVR'
        self.type_classifier = type_classifier  # Classifier to use: "SVM" or "SVR"
        self.seq_df_path = seq_df_path  # Path of csv file contained sequences informations
        self.num_test_videos = num_test_videos  # Number of videos of the dataset to considered to the clustering
        self.preliminary_clustering = preliminary_clustering  # Preliminary clustering performed
        self.classifier = None
        self.gmm_sequences = None
        self.fv_sequences = None
        self.vas_sequences = None

    """Determine the fisher vector cluster.
    Return cluster assigned to fisher vector """
    def __assignCluster(self, fisher_vector, n_kernels_preliminary_clustering):
        sum_max = cluster_max = 0
        for cluster in np.arange(n_kernels_preliminary_clustering):
            sum_cluster = sum(fisher_vector[cluster]) + sum(fisher_vector[cluster + n_kernels_preliminary_clustering])
            if sum_cluster > sum_max:
                sum_max=sum_cluster
                cluster_max=cluster
        return cluster_max

    """Generate multivector descriptor for each sequences. 
    Each sequence is characterized by a list containing the configurations relevant to the classification
    of the VAS index that occur during the sequence itself. 
    Return a list with the multivectors that describe the sequences of the num test videos of the dataset """
    def __get_multivector_sequences_descriptor(self, means_gmm_preliminary_clustering, relevant_configurations, fisher_vectors):
        print("---- Process multivector descriptor for sequences... ----")
        n_kernels_preliminary_clustering = len(means_gmm_preliminary_clustering)
        clusters_videos = []
        for seq_num in np.arange(self.num_test_videos):
            fisher_vector = fisher_vectors[seq_num][0]
            num_lndks = fisher_vector.shape[2]
            video_clusters = []
            for num_frame in np.arange(fisher_vector.shape[0]):
                frame_cluster = self.__assignCluster(fisher_vector[num_frame], n_kernels_preliminary_clustering)
                if frame_cluster in relevant_configurations:
                    video_clusters.append(means_gmm_preliminary_clustering[frame_cluster])
            clusters_videos.append(np.array(video_clusters).reshape(1, len(video_clusters), num_lndks))
        return clusters_videos

    """Group all the features into a single feature list to train the gaussian mixture.
    Return a 4D array that contains the features of all sequences """
    def __get_sequences_features(self,clusters_videos):
        total_num_frames = sum([video.shape[0] for video in clusters_videos])
        n_features_for_frame = clusters_videos[0].shape[2]
        data_videos_to_fit = np.ndarray(shape=(1, total_num_frames, 1, n_features_for_frame))
        index_frame = 0
        for video in clusters_videos:
            for index_video_frame in np.arange(video.shape[0]):
                current_frame_features = video[index_video_frame][0]
                for index_feature in np.arange(n_features_for_frame):
                    data_videos_to_fit[0][index_frame][0][index_feature] = current_frame_features[index_feature]
                index_frame += 1
        return data_videos_to_fit

    """ Train Gaussian Mixture suitable for describe the sequences.
        Return the fitted GMM """
    def __train_gmm_sequences(self, clusters_videos):
        print("---- Train GMM for sequences description... ----")
        video_to_considered=[vector for vector in clusters_videos if vector.shape[1] > 0]
        n_kernels_gmm = min([vector.shape[1] for vector in video_to_considered])
        return FisherVectorGMM(n_kernels=n_kernels_gmm).fit(self.__get_sequences_features(video_to_considered), n_init=10000)

    """Calculate the fisher vector for each sequence using GMM fitted and the multivector that describe the sequence. 
    Return a list contained the fisher vectors of all sequences """
    def __calculate_fv_sequences(self, clusters_videos):
        print("---- Calculate fisher vectors for sequences in dataset... ----")
        n_kernels_gmm = len(self.gmm_sequences.gmm.means_)
        fisher_vectors = []
        for num_video in np.arange(len(clusters_videos)):
            if clusters_videos[num_video].shape[1] > 0:
                fisher_vectors.append(self.gmm_sequences.predict(clusters_videos[num_video][:n_kernels_gmm])[0])
        return fisher_vectors

    """Read vas index of all sequences from dataset. 
    Return a list contained the vas index of all sequences """
    def __get_vas_videos_sequences(self, clusters_videos):
        print("---- Read vas indexes for sequences in dataset... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        vas_sequences = []
        for num_video in np.arange(len(clusters_videos)):
            if clusters_videos[num_video].shape[1] > 0:
                vas_sequences.append(seq_df.iloc[num_video][1])
        return vas_sequences

    """Reshape the fisher vectors transforming it from 3D in 2D to train classifier. 
    Return th fisher vectors in a 2D array """
    def __reshape_fv(self, fv_sequences):
        nsamples, nx, ny = fv_sequences.shape
        return fv_sequences.reshape((nsamples, nx * ny))

    """Train classifier using fisher vectors calculated and vas indexes readed of the sequences.
    The type of classifier (SVM or SVR) is passed to constructor of class.
    Return the trained classifier """
    def __train_classifier(self, percent_training_set, regularization_parameter, gamma_parameter):
        training_set_fv = self.fv_sequences[:int(percent_training_set * len(self.fv_sequences))]
        training_set_vas = self.vas_sequences[:int(percent_training_set * len(self.vas_sequences))]
        training_set_fv = self.__reshape_fv(np.array(training_set_fv))
        if self.type_classifier == "SVM":
            classifier = svm.SVC(C=regularization_parameter, gamma=gamma_parameter)
        else:
            classifier = svm.SVR(C=regularization_parameter, gamma=gamma_parameter)
        classifier.fit(training_set_fv, training_set_vas)
        return classifier

    def __init_data_sequences(self):

        gmm_preliminary_clustering = self.preliminary_clustering.gmm
        relevant_configurations = self.preliminary_clustering.index_relevant_configurations
        fisher_vectors = self.preliminary_clustering.fisher_vectors
        clusters_videos = self.__get_multivector_sequences_descriptor(gmm_preliminary_clustering.means,
                                                                      relevant_configurations, fisher_vectors)
        self.gmm_sequences = self.__train_gmm_sequences(clusters_videos)
        self.fv_sequences = self.__calculate_fv_sequences(clusters_videos)
        self.vas_sequences = self.__get_vas_videos_sequences(clusters_videos)

    def __train_classifier_maximizing_score(self, percent_training_set):
        print("Find parameters "+self.type_classifier+" that maximizes the total score... ")
        regularization_test_parameters = np.arange(10, 1010, 10)
        gamma_test_parameters = np.arange(0.1, 1.1, 0.1)
        max_rate = 0
        max_classifier = None
        for regularization in regularization_test_parameters:
            for gamma in gamma_test_parameters:
                self.classifier = self.__train_classifier(percent_training_set, regularization, gamma)
                current_rate = self.calculate_rate_model(percent_data_set=1 - percent_training_set)
                if current_rate > max_rate:
                    max_classifier = self.classifier
                    max_rate = current_rate
        return max_classifier

    """Performs the classifier training procedure based on what was done in the preliminary clustering phase"""
    def train_model(self, percent_training_set=0.85, regularization_parameter=1,
                    gamma_parameter='scale', train_by_max_score=True, classifier_dump_path=None):
        if self.fv_sequences == None or self.vas_sequences == None:
            self.__init_data_sequences()
        if train_by_max_score == True:
            self.classifier = self.__train_classifier_maximizing_score(percent_training_set=percent_training_set)
        else:
            self.classifier = self.__train_classifier(percent_training_set, regularization_parameter, gamma_parameter)
        if classifier_dump_path is not None:
            with open(classifier_dump_path, 'wb') as handle:
                pickle.dump(self.classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_rate_model(self, percent_data_set, path_scores_parameters=None):
        test_set_fv = self.fv_sequences[int((1-percent_data_set) * len(self.fv_sequences)):len(self.fv_sequences)]
        test_set_vas = self.vas_sequences[int((1-percent_data_set) * len(self.vas_sequences)):len(self.vas_sequences)]
        test_set_fv = self.__reshape_fv(np.array(test_set_fv))
        error = 0
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['video_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(test_set_fv.shape[0]):
            real_vas = test_set_vas[num_video]
            vas_predicted = self.classifier.predict(test_set_fv[num_video].reshape(1, -1))[0]
            error += abs(real_vas-vas_predicted)
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([num_video, real_vas, vas_predicted, abs(real_vas - vas_predicted)]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
        if path_scores_parameters is not None:
            out_df_scores.to_csv(path_scores_parameters, index=False, header=True)
        return 1/error if error > 0 else 1

    @staticmethod
    def load_model_from_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            model_classifier = pickle.load(f)
            assert isinstance(model_classifier, svm.SVC) or isinstance(model_classifier, svm.SVR)
        return model_classifier