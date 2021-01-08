import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import manifold


def check_existing_paths(dir_paths=[], file_paths=[]):
    project_path = os.getcwd() + "/"
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            os.mkdir(project_path + dir_path)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print("Configuration error: file '" + file_path + "' not exist in project")
            exit(1)


def get_subjects_seq_idx(seq_df_path):
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
    subject_idxs = get_subjects_seq_idx(seq_df_path)
    num_subject = len(subject_idxs)
    all_training_idx = []
    all_test_idx = []
    if cross_val_protocol == "Leave-One-Subject-Out":
        for subject_test in np.arange(num_subject):
            idxs_test = subject_idxs[subject_test]
            all_test_idx.append(np.array(idxs_test))
            all_training_idx.append(np.delete(np.arange(0, num_videos), idxs_test))
    elif cross_val_protocol == "5-fold-cross-validation":
        for subjects_test_offset in np.arange(0, num_subject, 5):
            idxs_test = []
            subjects_offset = subjects_test_offset + 5
            if subjects_offset >= len(subject_idxs):
                subjects_offset = len(subject_idxs)
            for subject_test in np.arange(subjects_test_offset, subjects_offset):
                idxs_test.append(subject_idxs[subject_test])
            idxs_test = sum(idxs_test, [])
            all_test_idx.append(np.array(idxs_test))
            all_training_idx.append(np.delete(np.arange(0, num_videos), idxs_test))
    elif cross_val_protocol == "Leave-One-Sequence-Out":
        for video_idx in np.arange(0, num_videos):
            all_test_idx.append(np.asarray([video_idx]))
            all_training_idx.append(np.delete(np.arange(0, num_videos), video_idx))

    return all_training_idx, all_test_idx


def plot_matrix(cm, labels, fname, normalize=True):
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


def save_data_on_csv(data, out_df, file_path):
    data_df_scores = np.hstack((np.array(data).reshape(1, -1)))
    out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                           ignore_index=True)
    out_df.to_csv(file_path, index=False, header=True)
    return out_df

def save_GMM_mean_info(gmm_means, selected_lndks_idx, csv_path, png_path):
    model = manifold.MDS(n_components=2, metric=True, n_init=4, random_state=1, max_iter=200, dissimilarity='euclidean')
    columns = ["#kernel"] + ["ldk #" + str(ldks_idx) for ldks_idx in selected_lndks_idx]
    out_gmm_means = pd.DataFrame(columns=[columns])
    for kernel_idx in np.arange(len(gmm_means)):
        data_gmm_means = np.hstack(
            (np.array([kernel_idx] + [center for center in gmm_means[kernel_idx]]).reshape(1, -1)))
        out_gmm_means = out_gmm_means.append(pd.Series(data_gmm_means.reshape(-1), index=out_gmm_means.columns),
                                             ignore_index=True)
    out_gmm_means.to_csv(csv_path, index=False, header=True)
    data_transformed = model.fit_transform(gmm_means)
    plt.plot(data_transformed[:, 0], data_transformed[:, 1], '.b')
    for k in np.arange(data_transformed.shape[0]):
        plt.annotate(str(k), (data_transformed[k, 0], data_transformed[k, 1]))
    plt.title('Position of %d clusters remapped in 2D with MSD' % (data_transformed.shape[0]))
    plt.savefig(png_path)
    plt.close()

def read_dict_from_csv(file_path, out_df, dict_labels):
    dict = {}
    if os.path.isfile(file_path):
        with open(file_path, 'r') as thresholds_rslt_file:
            reader = csv.reader(thresholds_rslt_file)
            for idx, row in enumerate(reader):
                if idx > 0:
                    dict[float(row[0])] = {}
                    for idx, label in enumerate(dict_labels):
                        dict[float(row[0])][label] = float(row[idx+1])
                    data = np.hstack((np.array([row[0], row[1], row[2]]).reshape(1, -1)))
                    out_df = out_df.append(pd.Series(data.reshape(-1), index=out_df.columns), ignore_index=True)
    return dict

def plot_graph(x, y, x_label, y_label, title, file_path, color = 'blue'):
    plt.plot(x, y, color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(file_path)
    plt.close()