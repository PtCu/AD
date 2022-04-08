import numpy as np
from pip import main
from sklearn.utils import check_random_state, check_array
import utilities.utils as utl
import os
import sys
import csv
import pandas as pd

cwd_path = os.getcwd()

source_data = cwd_path+"/data/simulated_data_source.tsv"

file_name = cwd_path+"/data/simulated_data2.tsv"
TARGET_NUM = 980


def read_data(filename, decimals=16):
    sys.stdout.write('\treading data...\n')
    feat_cov = None
    ID = None
    with open(filename) as f:
        data = list(csv.reader(f, delimiter='\t'))
        header = np.asarray(data[0])
        if 'GROUP' not in header:
            sys.stdout.write(
                'Error: group information not found. Please check csv header line for field "Group".\n')
            sys.exit(1)
        if 'ROI' not in header:
            sys.stdout.write(
                'Error: image features not found. Please check csv header line for field "IMG".\n')
            sys.exit(1)
        data = np.asarray(data[1:])

        group = (data[:, np.nonzero(header == 'GROUP')
                      [0]]).astype(int)
        feat_img = (data[:, np.nonzero(header == 'ROI')[0]]).astype(np.float)
        if 'COVAR' in header:
            feat_cov = (data[:, np.nonzero(header == 'COVAR')[0]]).astype(
                np.float)
        if 'ID' in header:
            ID = (data[:, np.nonzero(header == 'ID')[0]]).astype(np.int)
            # ID = ID[group == 1]

    return feat_cov, np.around(feat_img, decimals=decimals), ID, group


def get_data(filename, decimals=16):
    pt_nc_cov, pt_nc_img, ID, group = read_data(filename, decimals)

    # pt_nc_all = np.hstack((pt_nc_cov, pt_nc_img))

    return pt_nc_img, pt_nc_cov, ID, group


def bootstrap_sample_column(X, random_state=1234, n_samples=TARGET_NUM):
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
    if n_samples is None:
        n_samples = X.shape[0]

    return random_state.choice(X, size=n_samples, replace=True)


def uniform_sample_column(X, random_state=1234, n_samples=TARGET_NUM):
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
    if n_samples is None:
        n_samples = X.shape[0]

    min_X, max_X = np.min(X), np.max(X)
    return random_state.uniform(min_X, max_X, size=n_samples)


def generate_synthetic_features(X, method='bootstrap', random_state=1234, n_samples=TARGET_NUM):
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
    n_features = int(X.shape[1])
    # synth_X = np.empty_like(X)
    synth_X = np.zeros(shape=(n_samples, n_features))
    for column in range(n_features):
        # 每次填充一列
        if method == 'bootstrap':
            synth_X[:, column] = bootstrap_sample_column(
                X[:, column], random_state=random_state)
        elif method == 'uniform':
            synth_X[:, column] = uniform_sample_column(
                X[:, column], random_state=random_state)
        else:
            raise ValueError('method must be either `bootstrap` or `uniform`.')

    return synth_X


def generate_discriminative_dataset(X, method='bootstrap', random_state=1234, n_samples=TARGET_NUM):
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
        X, method=method, random_state=random_state)
    X_ = np.vstack((X, synth_X))
    y_ = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    permutation_indices = random_state.permutation(np.arange(X_.shape[0]))
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


if __name__ == "__main__":
    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(source_data)
    # pt_nc_all = gen_a_sample(ID, group, pt_nc_cov, pt_nc_img)
    pt_nc_all = np.hstack((ID, group, pt_nc_cov, pt_nc_img))
    X, Y = generate_discriminative_dataset(np.array(pt_nc_all))

    with open(file_name, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title = ["ID", "GROUP", "COVAR", "COVAR"]
        for i in range(0, 20):
            title.append("ROI")

        tsv_w.writerow(title)
    df = pd.DataFrame(X, columns=title)
    final_df = df.set_index("ID")

    # 保存 dataframe
    final_df.to_csv(file_name, sep='\t')
