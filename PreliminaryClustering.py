import pickle
import numpy as np
import pandas as pd
from FisherVectors import FisherVectorGMM
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

"""Class that is responsible for obtaining the relevant configurations for the classification of the VAS index. """
class PreliminaryClustering:
    def __init__(self,coord_df_path, seq_df_path, num_lndks, selected_lndks_idx, num_test_videos,n_kernels):
        self.coord_df_path = coord_df_path  # Path of csv file contained coordinates of the landmarks
        self.seq_df_path = seq_df_path  # Path of csv file contained sequences informations
        self.num_lndks = num_lndks  # Number of landmarks for each frame of the videos in the dataset
        self.selected_lndks_idx = selected_lndks_idx  # Indexes of the landmarks to considered to the clustering
        self.num_test_videos = num_test_videos  # Number of videos of the dataset to considered to the clustering
        self.n_kernels = n_kernels  # Number of kernels of the gmm to trained
        self.gmm = None
        self.fisher_vectors = None
        self.histograms_of_videos = None
        self.index_relevant_configurations = None
        self.index_neutral_configurations = None

    """ Extract velocities of landmarks video sequences in dataset. 
    Return a list of 2D array with velocities of the landmarks for each frame """
    def __get_velocities_frames(self):
        print("---- Calculating velocities of the frames in dataset... ----")
        coord_df = pd.read_csv(self.coord_df_path)
        seq_df = pd.read_csv(self.seq_df_path)
        velocities = []
        for seq_num in np.arange(seq_df.shape[0]):
            lndks = coord_df.loc[coord_df['0']==seq_num].values
            lndks = lndks[:,2:]
            nose_tip_x = lndks[:,30]
            nose_tip_y = lndks[:,30+self.num_lndks]
            offset = np.hstack( (np.repeat(nose_tip_x.reshape(-1,1), self.num_lndks, axis=1),
                                 np.repeat(nose_tip_y.reshape(-1,1), self.num_lndks, axis=1) ) )
            lndks_centered = lndks - offset
            lndk_vel = np.power(np.power(lndks_centered[0:lndks_centered.shape[0]-1,0:self.num_lndks] -
                                         lndks_centered[1:lndks_centered.shape[0],0:self.num_lndks], 2) +
                                np.power(lndks_centered[0:lndks_centered.shape[0]-1,self.num_lndks:] -
                                         lndks_centered[1:lndks_centered.shape[0],self.num_lndks:], 2),
                                0.5)
            data_velocities=[]
            for k in np.arange(lndk_vel.shape[0]):
                data_velocities.append(np.array(lndk_vel[k, self.selected_lndks_idx]).reshape(1, -1))
            velocities.append(np.array(data_velocities))
        return velocities

    """ Prepare features for GMM training.
    All velocities of the sequences frame are inserted in a 4D array that contains all frames informations.
    Features of the frames of all videos are collected in the same sequence.
    Return a 4D array with velocities of the landmarks for each frame in the dataset """
    def __get_videos_frames_features(self, velocities):
        print("---- Get features vector of the frame in dataset by velocities... ----")
        total_num_frames = sum([video.shape[0] for video in velocities])
        n_features_for_frame = velocities[0].shape[2]
        data_videos_to_fit = np.ndarray(shape=(1, total_num_frames, 1, n_features_for_frame))
        index_frame = 0
        for video in velocities:
            for index_video_frame in np.arange(video.shape[0]):
                current_frame_features = video[index_video_frame][0]
                for index_feature in np.arange(n_features_for_frame):
                    data_videos_to_fit[0][index_frame][0][index_feature] = current_frame_features[index_feature]
                index_frame += 1
        return data_videos_to_fit

    """ Train Gaussian Mixture for process fisher vectors.
    Return the fitted GMM """
    def __generate_gmm(self, videos_features):
        print("---- Generate GMM with " + str(self.n_kernels) + " kernels... ----")
        return FisherVectorGMM(n_kernels=self.n_kernels).fit(videos_features)

    """ Calculate the fisher vectors of the first num test videos of the dataset.
    Return the calculated fisher vectors """
    def __calculate_FV(self, velocities):
        print("---- Calculate fisher vectors of video sequences in dataset... ----")
        fisher_vectors = []
        n_features_for_frame = velocities[0].shape[2]
        for i in range(0, self.num_test_videos):
            fv = self.gmm.predict(np.array(velocities[i]).reshape(1, velocities[i].shape[0], 1, n_features_for_frame))
            fisher_vectors.append(fv)
        return fisher_vectors

    """Determine the fisher vector cluster and update the histogram.
    Return the updated histogram"""
    def __clustering(self, fv, histogram):
        sum_max = cluster_max=0
        for cluster in np.arange(self.n_kernels):
            sum_cluster = sum(fv[cluster])+sum(fv[cluster+self.n_kernels])
            if sum_cluster > sum_max:
                sum_max = sum_cluster
                cluster_max = cluster
        histogram[cluster_max] += 1
        return histogram

    """ Calculate the histograms of the videos starting from the fisher vectors of the frames.
    Uses the clustering method to establish the cluster of a sequence by his fisher vector. 
    Return a list with histograms of videos """
    def __generate_histograms(self):
        print("---- Generate histograms of video sequences... ----")
        n_videos = len(self.fisher_vectors)
        histograms_of_videos = []
        for index in range(0, n_videos):
            current_video_fv = self.fisher_vectors[index][0]
            video_histogram = np.zeros(self.n_kernels)
            for i in range(0, current_video_fv.shape[0]):
                video_histogram=self.__clustering(current_video_fv[i],video_histogram)
            video_histogram = video_histogram / sum(video_histogram)
            histograms_of_videos.append(video_histogram)
        return histograms_of_videos

    """ Apply a strategy to derive the relevant and neutral configurations for classify the VAS index using histograms.
    Return two lists containing respectively the indices of the relevant and neautral configurations to classify 
    the vas index """
    def __generate_relevant_and_neutral_configurations(self, threshold_neutral):
        print("---- Process relevant and neutral configurations... ----")
        seq_df = pd.read_csv(self.seq_df_path)
        index_neutral_configurations  = []
        for seq_num in np.arange(self.num_test_videos):
            vas = seq_df.iloc[seq_num][1]
            hist = self.histograms_of_videos[seq_num]
            if vas == 0:
                for j in np.arange(self.n_kernels):
                    if hist[j] > threshold_neutral and j not in index_neutral_configurations:
                        index_neutral_configurations.append(j)
        index_relevant_configurations = [x for x in np.arange(self.n_kernels) if x not in index_neutral_configurations]
        return index_relevant_configurations , index_neutral_configurations

    """ Plot and save histograms by distinguishing the color of the representations of the relevant configurations
    from the neutral ones """
    def __plot_and_save_histograms(self, histo_figures_path):
        for i in range(0, len(self.histograms_of_videos)):
            print("Plot and save histogram #"+str(i)+"...")
            histo = self.histograms_of_videos[i]
            plt.bar(self.index_neutral_configurations, histo[np.array(self.index_neutral_configurations)], color="blue")
            plt.bar(self.index_relevant_configurations, histo[np.array(self.index_relevant_configurations)], color="red")
            plt.title("VIDEO #"+str(i))
            plt.savefig(histo_figures_path+'/video-%03d.png' %i, dpi=200)
            plt.close()

    """ Execute preliminary clustering using the parameters passed to class constructor.
    If plot_and_save_histo is setted on True value the figures of histograms of videos is saved in files """
    def execute_preliminary_clustering(self, threshold_neutral=0.015, preliminary_clustering_dump_path=None, histo_figures_path=None,
                                       plot_and_save_histo=False):
        velocities = self.__get_velocities_frames() #Velocities of landmarks for each frame of the videos content in the dataset
        if self.gmm == None:
            data_video_to_fit = self.__get_videos_frames_features(velocities)  # Velocities of landmarks collected in the same 4D array
            self.gmm = self.__generate_gmm(data_video_to_fit) # Gaussian mixture that performs the clustering of the configurations
        if self.fisher_vectors == None:
            self.fisher_vectors = self.__calculate_FV(velocities) # Fisher vectors for each frame of the videos contained in the dataset
        if self.histograms_of_videos == None:
            self.histograms_of_videos = self.__generate_histograms() # Histograms of the configurations detected for each frame of the videos contained in the dataset
        self.index_relevant_configurations, self.index_neutral_configurations = \
            self.__generate_relevant_and_neutral_configurations(threshold_neutral) # Relevant and neutral configurations for the classification of the vas index
        if plot_and_save_histo:
            self.__plot_and_save_histograms(histo_figures_path)
        if preliminary_clustering_dump_path is not None:
            with open(preliminary_clustering_dump_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            preliminary_clustering = pickle.load(f)
            assert isinstance(preliminary_clustering, PreliminaryClustering)
        return preliminary_clustering