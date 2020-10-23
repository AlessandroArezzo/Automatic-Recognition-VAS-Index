import pandas as pd
import pickle
import numpy as np
from sklearn import svm

"""Class that deals with training a classifier (SVM or SVR) starting from the relevant configurations 
to characterize the VAS index obtained during the preliminary clustering phase. """
class ModelClassifier:

    def __init__(self, seq_df_path, train_video_idx, test_video_idx, preliminary_clustering, type_classifier='SVM',
                 verbose=True):
        assert type_classifier == 'SVM' or type_classifier == 'SVR'
        self.type_classifier = type_classifier  # Classifier to use: "SVM" or "SVR"
        self.seq_df_path = seq_df_path  # Path of csv file contained sequences informations
        self.train_video_idx = train_video_idx  # Indexes of the videos to use for training
        self.test_video_idx = test_video_idx  # Indexes of the videos to use for test
        self.preliminary_clustering = preliminary_clustering  # Preliminary clustering performed
        self.classifier = None
        self.vas_sequences = None
        self.histo_relevant_config_videos = None
        self.means_gmm = self.preliminary_clustering.gmm.means
        self.dict_relevant_config = {}
        self.verbose = verbose
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
            sum_histo_values = sum(histo_relevant_config)
            if sum_histo_values > 0:
                histo_relevant_config = histo_relevant_config / sum_histo_values
            histo_relevant_config_videos.append(histo_relevant_config)
        self.histo_relevant_config_videos = histo_relevant_config_videos


    """Read vas index of all sequences from dataset. 
    Return a list contained the vas index of all sequences """
    def __read_vas_videos(self):
        if self.verbose:
            print("---- Read vas indexes for sequences in dataset... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        vas_sequences = []
        for num_video in np.arange(len(self.histo_relevant_config_videos)):
            vas_sequences.append(seq_df.iloc[num_video][1])
        self.vas_sequences = vas_sequences

    """Train classifier using fisher vectors calculated and vas indexes readed of the sequences.
    The type of classifier (SVM or SVR) is passed to constructor of class.
    Return the trained classifier """
    def __train_classifier(self, regularization_parameter, gamma_parameter):
        training_set_histo = np.asarray([self.histo_relevant_config_videos[i] for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        if self.type_classifier == "SVM":
            classifier = svm.SVC(C=regularization_parameter, gamma=gamma_parameter)
        else:
            classifier = svm.SVR(C=regularization_parameter, gamma=gamma_parameter)
        classifier.fit(training_set_histo, training_set_vas)
        return classifier

    def __train_classifier_maximizing_score(self):
        if self.verbose:
            print("---- Find parameters "+self.type_classifier+" that maximizes the total score on the test sequences... ----")
        regularization_test_parameters = np.arange(10, 1010, 10)
        gamma_test_parameters = np.arange(0.1, 1.1, 0.1)
        min_error = np.inf
        best_classifier = None
        for regularization in regularization_test_parameters:
            for gamma in gamma_test_parameters:
                self.classifier = self.__train_classifier(regularization, gamma)
                current_error = self.calculate_rate_model()[0]
                if current_error < min_error:
                    best_classifier = self.classifier
                    min_error = current_error
        self.classifier = best_classifier

    def __init_data_sequences(self):
        self.__generate_histo_relevant_configuration()
        self.__read_vas_videos()

    """Performs the classifier training procedure based on what was done in the preliminary clustering phase"""
    def train_model(self, regularization_parameter=1,
                    gamma_parameter='scale', train_by_max_score=True, classifier_dump_path=None):
        if self.histo_relevant_config_videos == None or self.vas_sequences == None:
            self.__init_data_sequences()
        if train_by_max_score == True:
            self.classifier = self.__train_classifier_maximizing_score()
        else:
            self.classifier = self.__train_classifier(regularization_parameter, gamma_parameter)
        if classifier_dump_path is not None:
            with open(classifier_dump_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def calculate_rate_model(self, path_scores_parameters=None):
        test_set_histo = np.asarray([self.histo_relevant_config_videos[i] for i in self.test_video_idx])
        test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
        sum_error = 0
        count_corrected_predict = 0
        num_test_videos = test_set_histo.shape[0]
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['sequence_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(num_test_videos):
            real_vas = test_set_vas[num_video]
            vas_predicted = self.classifier.predict(test_set_histo[num_video].reshape(1,-1))[0]
            vas_predicted = round(vas_predicted, 3)
            error = abs(real_vas-vas_predicted)
            sum_error += error
            error = round(error, 3)
            if error < 0.5:
                count_corrected_predict += 1
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([self.test_video_idx[num_video], real_vas, vas_predicted, error]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
        if path_scores_parameters is not None:
            out_df_scores.to_csv(path_scores_parameters, index=False, header=True)
        sum_error /= num_test_videos
        sum_error = round(sum_error, 3)
        accuracy = round((count_corrected_predict / num_test_videos)*100, 2)
        return sum_error, accuracy

    @staticmethod
    def load_model_from_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            model_classifier = pickle.load(f)
            assert isinstance(model_classifier, ModelClassifier)
        return model_classifier