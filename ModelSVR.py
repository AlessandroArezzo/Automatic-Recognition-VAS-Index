import pandas as pd
import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from utils import plotMatrix

"""Class that deals with training an SVR starting from the relevant configurations 
to characterize the VAS index obtained during the preliminary clustering phase. """
class ModelSVR:

    def __init__(self, seq_df_path, train_video_idx, test_video_idx, preliminary_clustering, weighted_samples=False, verbose=True):
        self.seq_df_path = seq_df_path  # Path of csv file contained sequences informations
        self.train_video_idx = train_video_idx  # Indexes of the videos to use for training
        self.test_video_idx = test_video_idx  # Indexes of the videos to use for test
        self.preliminary_clustering = preliminary_clustering  # Preliminary clustering performed
        self.classifier = None
        self.vas_sequences = None
        self.desc_relevant_config_videos = None
        self.means_gmm = self.preliminary_clustering.gmm.means
        self.dict_relevant_config = {}
        self.verbose = verbose
        self.weighted_samples = weighted_samples
        index = 0
        self.sample_weights = None
        for config in self.preliminary_clustering.index_relevant_configurations:
            mean_gmm = self.means_gmm[config]
            self.dict_relevant_config[str(mean_gmm)] = index
            index += 1

    def __generate_descriptors_relevant_configuration(self):
        if self.verbose:
            print("---- Generate descriptors of video sequences... ----")
        fisher_vectors = self.preliminary_clustering.fisher_vectors
        n_videos = len(fisher_vectors)
        num_relevant_config = len(self.preliminary_clustering.index_relevant_configurations)
        descriptors_of_videos = []
        idx_relevant_frame_mean = self.preliminary_clustering.index_relevant_configurations
        idx_relevant_frame_sd = [config + len(self.means_gmm) for config in self.preliminary_clustering.index_relevant_configurations]
        means_gmm = self.means_gmm[self.preliminary_clustering.index_relevant_configurations]
        idx_relevant_config_mean = [self.dict_relevant_config[str(mean_gmm)] for mean_gmm in means_gmm]
        idx_relevant_config_sd = [self.dict_relevant_config[str(mean_gmm)] + num_relevant_config for mean_gmm in means_gmm]
        for index in range(0, n_videos):
            current_video_fv = fisher_vectors[index][0]
            video_descriptor = np.zeros(shape=(num_relevant_config * 2, current_video_fv.shape[2]))
            for index_frame in range(0, current_video_fv.shape[0]):
                frame = current_video_fv[index_frame]
                video_descriptor[idx_relevant_config_mean] += frame[idx_relevant_frame_mean]
                video_descriptor[idx_relevant_config_sd] += frame[idx_relevant_frame_sd]
            if sum(video_descriptor).any() != 0:
                video_descriptor = np.sqrt(np.abs(video_descriptor)) * np.sign(video_descriptor)
                video_descriptor = video_descriptor / np.linalg.norm(video_descriptor, axis=(0, 1))[None, None]
            descriptors_of_videos.append(video_descriptor)
        self.desc_relevant_config_videos = descriptors_of_videos

    """Read vas index of all sequences from dataset. 
    Return a list contained the vas index of all sequences """
    def __read_vas_videos(self):
        if self.verbose:
            print("---- Read vas indexes for sequences in dataset... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        vas_sequences = []
        for num_video in np.arange(len(self.desc_relevant_config_videos)):
            vas_sequences.append(seq_df.iloc[num_video][1])
        self.vas_sequences = vas_sequences

    """Train SVR using fisher vectors calculated and vas indexes readed of the sequences.
    Return the trained classifier """
    def __train_SVR(self, regularization_parameter, gamma_parameter):
        training_set_histo = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        model_svr = svm.SVR(C=regularization_parameter, gamma=gamma_parameter)
        return model_svr.fit(training_set_histo, training_set_vas)

    def __train_SVR_maximizing_score(self, n_jobs):
        if self.verbose:
            print("---- Find parameters that minimizes mean absolute error... ----")
        training_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.train_video_idx])
        training_set_vas = np.asarray([self.vas_sequences[i] for i in self.train_video_idx])
        param = {'kernel': ['rbf'], 'C': np.arange(1, 100, 10),
                 'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}
        grid_result = GridSearchCV(estimator=svm.SVR(), param_grid=param, scoring='neg_mean_absolute_error',
                             n_jobs=n_jobs).fit(training_set_desc, training_set_vas, sample_weight=self.sample_weights)
        best_params = grid_result.best_params_
        return svm.SVR(kernel=best_params["kernel"], C=best_params["C"],
                          gamma=best_params["gamma"]).fit(training_set_desc, training_set_vas, sample_weight=self.sample_weights)

    def __init_data_sequences(self):
        self.__generate_descriptors_relevant_configuration()
        self.__read_vas_videos()

    """Performs the model training procedure based on what was done in the preliminary clustering phase"""
    def train_SVR(self, regularization_parameter=1,
                    gamma_parameter='scale', train_by_max_score=True, classifier_dump_path=None, n_jobs=1):
        if self.desc_relevant_config_videos == None or self.vas_sequences == None:
            self.__init_data_sequences()
        if self.weighted_samples:
            self.__calculate_sample_weights()
        if train_by_max_score == True:
            self.classifier = self.__train_SVR_maximizing_score(n_jobs=n_jobs)
        else:
            self.classifier = self.__train_SVR(regularization_parameter, gamma_parameter)
        if classifier_dump_path is not None:
            with open(classifier_dump_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def calculate_rate_model(self, path_scores_parameters=None, path_scores_cm=None):
        test_set_desc = np.asarray([self.desc_relevant_config_videos[i].flatten() for i in self.test_video_idx])
        test_set_vas = np.asarray([self.vas_sequences[i] for i in self.test_video_idx])
        sum_error = 0
        num_test_videos = test_set_desc.shape[0]
        cm = np.zeros(shape=(11, 11))
        real_all_vas = []
        predicted_all_vas = []
        if path_scores_parameters is not None:
            out_df_scores = pd.DataFrame(columns=['sequence_num', 'real_vas', 'vas_predicted', 'error'])
        for num_video in np.arange(num_test_videos):
            real_vas = test_set_vas[num_video]
            real_all_vas.append(real_vas)
            vas_predicted = self.classifier.predict(test_set_desc[num_video].reshape(1,-1))[0]
            if vas_predicted < 0:
                vas_predicted = 0
            elif vas_predicted > 10:
                vas_predicted = 10
            vas_predicted = int(round(vas_predicted, 0))
            predicted_all_vas.append(vas_predicted)
            error = abs(real_vas-vas_predicted)
            sum_error += error
            if path_scores_parameters is not None:
                data = np.hstack(
                    (np.array([self.test_video_idx[num_video], real_vas, vas_predicted, error]).reshape(1, -1)))
                out_df_scores = out_df_scores.append(pd.Series(data.reshape(-1), index=out_df_scores.columns),
                                                     ignore_index=True)
            cm[real_vas][vas_predicted] += 1
        if path_scores_parameters is not None:
            out_df_scores.to_csv(path_scores_parameters, index=False, header=True)
        if path_scores_cm is not None:
            plotMatrix(cm=cm, labels=np.arange(0, 11), normalize=True, fname=path_scores_cm)
        mean_error = sum_error / num_test_videos
        return round(mean_error, 3), cm

    def __calculate_sample_weights(self):
        vas_occ = {}
        vas_weights = {}
        for vas in np.arange(0, 11):
            vas_occ[vas] = [self.vas_sequences[i] for i in self.train_video_idx].count(vas)
        sum_vas_occ = sum(vas_occ.values())
        for vas in np.arange(0, 11):
            if vas_occ[vas] > 0:
                vas_weights[vas] = sum_vas_occ / (11 * vas_occ[vas])
            else:
                vas_weights[vas] = 0
        self.sample_weights = np.ones(len(self.train_video_idx))
        for idx, video_idx in enumerate(self.train_video_idx):
            self.sample_weights[idx] = vas_weights[self.vas_sequences[video_idx]]

    @staticmethod
    def load_model_from_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            model_classifier = pickle.load(f)
            assert isinstance(model_classifier, ModelSVR)
        return model_classifier