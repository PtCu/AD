import numpy as np
from pip import main
from pyparsing import col
from sklearn.utils import check_random_state, check_array
import utilities.utils as utl
import os
import sys
import csv
import pandas as pd

cwd_path = os.getcwd()
data_dir = cwd_path+"/data/"

EXTRA_DATA_NUM = 500

source_file1 = os.path.join(data_dir, "1_NC001_ts.csv")
source_file2 = os.path.join(data_dir, "1_NC002_ts.csv")
source_file3 = os.path.join(data_dir, "1_NC003_ts.csv")
source_file4 = os.path.join(data_dir, "1_NC004_ts.csv")
tmp_file = os.path.join(data_dir, "tmp_data.csv")
dest_file1 = os.path.join(data_dir, "synthetic_data1.csv")
dest_file2 = os.path.join(data_dir, "synthetic_data2.csv")
data = []

SAMPLE_NUM = 170
STEP = 50
GROUP_LABEL = 1
# 正常人标签
NC_TYPE = 0
# 病号标签
PT_TYPE = 1

# title1 = ["ID", "SET", "GROUP"]
# title2 = ["participant_id", "session_id", "diagnosis"]

title1 = ["SET", "GROUP"]
title2 = ["session_id", "diagnosis"]

source_file_list = [source_file1, source_file2, source_file3, source_file4]

session_id = 0


def read_data():
    for idx in range(len(source_file_list)):
        df = pd.read_csv(source_file_list[idx], header=None, sep=',')
        if idx == 0:
            for i in range(df.shape[1]):
                title1.append("ROI")
                title2.append("ROI"+str(i+1))
        i = 0
        j = SAMPLE_NUM-1
        id = 0
        while j < df.shape[1]:
            for r in range(i, j):
                item = []
                # SET
                item.append(session_id)
                # GROUP
                if idx == 0:
                    item.append(NC_TYPE)
                else:
                    item.append(PT_TYPE)
                for l in range(df.shape[1]):
                    item.append(df.at[r, l])
                data.append(item)
            i += STEP
            j += STEP


def bootstrap_sample_column(X, n_samples, random_state=1234):
    """bootstrap_sample_column

    Bootstrap sample the column of a dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to bootstrap

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        The bootstrapped column.
    """
    random_state = check_random_state(random_state)

    return random_state.choice(X, size=n_samples, replace=True)


def uniform_sample_column(X, n_samples, random_state=1234):
    """uniform_sample_column

    Sample a column uniformly between its minimum and maximum value.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to sample.

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        Uniformly sampled column.
    """
    random_state = check_random_state(random_state)

    min_X, max_X = np.min(X), np.max(X)
    return random_state.uniform(min_X, max_X, size=n_samples)


def generate_synthetic_features(X, n_samples, method='bootstrap', random_state=1234):
    """generate_synthetic_features

    Generate a synthetic dataset based on the empirical distribution
    of `X`.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    synth_X : np.ndarray (n_samples, n_features)
        The synthetic dataset.
    """
    random_state = check_random_state(random_state)
    n_features = len(X[0])
    # synth_X = np.empty_like(X)
    synth_X = np.zeros(shape=(n_samples, n_features))
    for column in range(n_features):
        # 每次填充一列
        if method == 'bootstrap':
            synth_X[:, column] = bootstrap_sample_column(
                X[:, column], n_samples, random_state=random_state)
        elif method == 'uniform':
            synth_X[:, column] = uniform_sample_column(
                X[:, column], n_samples, random_state=random_state)
        else:
            raise ValueError('method must be either `bootstrap` or `uniform`.')

    return synth_X


def generate_discriminative_dataset(X,  n_samples, method='bootstrap', random_state=1234):
    """generate_discriminative_dataset.

    Generate a synthetic dataset based on the empirical distribution
    of `X`. A target column will be returned that is 0 if the row is
    from the real distribution, and 1 if the row is synthetic. The
    number of synthetic rows generated is equal to the number of rows
    in the original dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    X_ : np.ndarray (2 * n_samples, n_features)
        Feature array for the synthetic dataset. The rows
        are randomly shuffled, so synthetic and actual samples should
        be intermixed.

    y_ : np.ndarray (2 * n_samples)
        Target column indicating whether the row is from the actual
        dataset (0) or synthetic (1).
    """
    random_state = check_random_state(random_state)

    synth_X = generate_synthetic_features(
        X, n_samples, method=method, random_state=random_state)
    X_ = np.vstack((X, synth_X))
    y_ = np.concatenate((np.ones(len(X)), np.zeros(len(synth_X))))

    permutation_indices = random_state.permutation(np.arange(len(X_)))
    X_ = X_[permutation_indices, :]
    y_ = y_[permutation_indices]

    return X_, y_


def gen_a_sample(ID, group, pt_nc_cov, pt_nc_img):
    data = []
    for i in range(len(ID)):
        item = []
        item.append(ID[i][0])
        item.append(group[i][0])
        for j in range(len(pt_nc_cov[i])):
            item.append(pt_nc_cov[i][j])
        for j in range(len(pt_nc_img[i])):
            item.append(pt_nc_img[i][j])

        data.append(item)
    return data


def add_label(X, title, dest_file):

    df = pd.DataFrame(X, columns=title)
    if "SET" in title:
        id_name = "ID"
    else:
        id_name = "participant_id"
    df.index.name = id_name
    df.to_csv(dest_file, sep='\t')
    return



if __name__ == "__main__":
    read_data()

    X, Y = generate_discriminative_dataset(
        np.array(data), EXTRA_DATA_NUM)
    add_label(X, title1, dest_file1)
    add_label(X, title2, dest_file2)
