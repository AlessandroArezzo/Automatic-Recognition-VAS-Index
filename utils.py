import os
import numpy as np

def check_existing_paths(dir_paths=[],file_paths=[]):
    project_path = os.getcwd()+"/"
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(project_path+dir_path)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print("Configuration error: file '" + file_path + "' not exist in project")
            exit(1)

def get_training_and_test_idx(num_videos, cross_val_protocol):
    all_training_idx = []
    all_test_idx = []
    if cross_val_protocol == "Leave-One-Sequence-Out":
        for video_idx in np.arange(0, num_videos):
            all_test_idx.append(np.array([video_idx]))
            all_training_idx.append(np.delete(np.arange(0,num_videos),video_idx))
    elif cross_val_protocol == "5-fold-cross-validation":
        for video_idx in np.arange(0, num_videos, 5):
            test_idx_round = np.arange(0, num_videos)[video_idx:video_idx+5]
            all_test_idx.append(test_idx_round)
            all_training_idx.append(np.delete(np.arange(0,num_videos),test_idx_round))
    return all_training_idx, all_test_idx