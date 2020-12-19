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

def get_subjects_seq_ids(seq_df_path):
    seq_df = pd.read_csv(seq_df_path)
    subjects_idxs = {}
    subject_count = 0
    seq_name = ""
    for seq_num in np.arange(seq_df.shape[0]):
        if seq_num != 0 and seq_name[0:6] != seq_df.iloc[seq_num][0][0:6]:
            subject_count += 1
        if subject_count not in subjects_idxs:
            subjects_idxs[subject_count] = []
        subjects_idxs[subject_count].append(seq_num)
        seq_name = seq_df.iloc[seq_num][0]
    return subjects_idxs


def get_training_and_test_idx(num_videos, cross_val_protocol, seq_df_path):
    subject_idxs = get_subjects_seq_ids(seq_df_path)
    num_subject = len(subject_idxs)
    all_training_idx = []
    all_test_idx = []
    if cross_val_protocol == "Leave-One-Sequence-Out":
        for subject_test in np.arange(num_subject):
            idxs_test = subject_idxs[subject_test]
            all_test_idx.append(np.array(idxs_test))
            all_training_idx.append(np.delete(np.arange(0,num_videos), idxs_test))

    elif cross_val_protocol == "5-fold-cross-validation":
        for subjects_test_offset in np.arange(0, num_subject, 5):
            idxs_test = []
            for subject_test in np.arange(subjects_test_offset, subjects_test_offset + 5):
                idxs_test.append(subject_idxs[subject_test])
            idxs_test = sum(idxs_test, [])
            all_test_idx.append(np.array(idxs_test))
            all_training_idx.append(np.delete(np.arange(0, num_videos), idxs_test))
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