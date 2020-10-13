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
        self.vas_sequences = None
        self.histo_relevant_config_videos = None
        self.means_gmm = self.preliminary_clustering.gmm.means
        self.dict_relevant_config = {}
        index = 0
        for config in self.preliminary_clustering.index_relevant_configurations:
            mean_gmm = self.means_gmm[config]
            self.dict_relevant_config[str(mean_gmm)] = index
            index += 1

    def __generate_histo_relevant_configuration(self):
        histo_of_videos = self.preliminary_clustering.histograms_of_videos
        index_relevant_configuration = self.preliminary_clustering.index_relevant_configurations
        histo_relevant_config_videos = []
        for histo in histo_of_videos:
            histo_relevant_config = np.zeros(len(index_relevant_configuration))
            for config in index_relevant_configuration:
                mean_gmm = self.means_gmm[config]
                histo_relevant_config[self.dict_relevant_config[str(mean_gmm)]] = histo[config]
            histo_relevant_config = histo_relevant_config / sum(histo_relevant_config)
            histo_relevant_config_videos.append(histo_relevant_config)
        self.histo_relevant_config_videos = histo_relevant_config_videos


    """Read vas index of all sequences from dataset. 
    Return a list contained the vas index of all sequences """
    def __read_vas_videos_sequences(self):
        print("---- Read vas indexes for sequences in dataset... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        vas_sequences = []
        for num_video in np.arange(self.num_test_videos):
            vas_sequences.append(seq_df.iloc[num_video][1])
        self.vas_sequences = vas_sequences

    """Train classifier using fisher vectors calculated and vas indexes readed of the sequences.
    The type of classifier (SVM or SVR) is passed to constructor of class.
    Return the trained classifier """
    def __train_classifier(self, percent_training_set, regularization_parameter, gamma_parameter):
        training_set_histo = self.histo_relevant_config_videos[:int(percent_training_set
                                                                    * len(self.histo_relevant_config_videos))]
        training_set_vas = self.vas_sequences[:int(percent_training_set * len(self.vas_sequences))]
        training_set_histo = np.asarray(training_set_histo)
        if self.type_classifier == "SVM":
            classifier = svm.SVC(C=regularization_parameter, gamma=gamma_parameter)
        else:
            classifier = svm.SVR(C=regularization_parameter, gamma=gamma_parameter)
        classifier.fit(training_set_histo, training_set_vas)
        return classifier

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

    def __init_data_sequences(self):
        self.__generate_histo_relevant_configuration()
        self.__read_vas_videos_sequences()

    """Performs the classifier training procedure based on what was done in the preliminary clustering phase"""
    def train_model(self, percent_training_set=0.85, regularization_parameter=1,
                    gamma_parameter='scale', train_by_max_score=True, classifier_dump_path=None):
        if self.histo_relevant_config_videos == None or self.vas_sequences == None:
            self.__init_data_sequences()
        if train_by_max_score == True:
            self.classifier = self.__train_classifier_maximizing_score(percent_training_set=percent_training_set)
        else:
            self.classifier = self.__train_classifier(percent_training_set, regularization_parameter, gamma_parameter)
        if classifier_dump_path is not None:
            with open(classifier_dump_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_rate_model(self, percent_data_set, path_scores_parameters=None):
        test_set_histo = self.histo_relevant_config_videos[int((1-percent_data_set)
                                                               * len(self.histo_relevant_config_videos)):len(
                                                                    self.histo_relevant_config_videos)]
        test_set_vas = self.vas_sequences[int((1-percent_data_set) * len(self.vas_sequences)):len(self.vas_sequences)]
        test_set_histo = np.asarray(test_set_histo)
        error = 0
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['video_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(test_set_histo.shape[0]):
            real_vas = test_set_vas[num_video]
            vas_predicted = self.classifier.predict(test_set_histo[num_video].reshape(1,-1))[0]
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
            assert isinstance(model_classifier, ModelClassifier)
        return model_classifier