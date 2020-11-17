import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

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

def plotMatrix(cm, labels, fname, normalize=True):
    # Normalize confusion matrix
    if normalize:
        for row_idx in np.arange(cm.shape[0]):
            sum_row = sum(cm[row_idx])
            if sum_row > 0:
                cm[row_idx] = cm[row_idx] / sum_row
    df_cm = pd.DataFrame(cm, index=[str(i) for i in labels],
                         columns=[str(i) for i in labels])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.savefig(fname, dpi=240)
    plt.close()