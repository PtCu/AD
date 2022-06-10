import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from HYDRA.mlni.utils import consensus_clustering, cv_cluster_stability, hydra_solver_svm, time_bar, cluster_stability
from HYDRA.mlni.base import WorkFlow
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn import decomposition
 
rng = np.random.RandomState(1)
__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"
cwd_path = os.getcwd()
output_dir = cwd_path + "/HYDRA/output/"


class RB_DualSVM_Subtype(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for clustering.
    """

    def __init__(self, input, feature_tsv, split_index, cv_repetition, k_min, k_max, output_dir, label, balanced=True,
                 n_iterations=100, test_size=0.2, num_consensus=20, num_iteration=50, tol=1e-6, predefined_c=None,
                 weight_initialization_type='DPP', n_threads=8, save_models=False, verbose=False):

        self._input = input
        self._feature_tsv = feature_tsv
        self._split_index = split_index
        self._cv_repetition = cv_repetition
        self._n_iterations = n_iterations
        self._output_dir = output_dir
        self._k_min = k_min
        self._k_max = k_max
        self._balanced = balanced
        self._test_size = test_size
        self._num_consensus = num_consensus
        self._num_iteration = num_iteration
        self._tol = tol
        self._predefined_c = predefined_c
        self._weight_initialization_type = weight_initialization_type
        self._k_range_list = list(range(k_min, k_max + 1))
        self._n_threads = n_threads
        self._save_models = save_models
        self._verbose = verbose
        self._label = label
        self.repeat_num = 5

    def normalization(self, data):
        data = np.array(data)
        _range = np.max(abs(data))
        return data / _range

    def get_K_from_three_score(self, sh_y, ch_y, db_y):
        sh_y = np.array(sh_y)
        ch_y = np.array(ch_y)
        db_y = np.array(db_y)
        sh_k = np.argmax(sh_y)
        ch_k = np.argmax(ch_y)
        db_k = np.argmin(db_y)
        # 三者都一样
        if sh_k == ch_k and sh_k == db_k:
            return sh_k+2
        # 两个一样
        elif db_k == ch_k:
            return db_k+2
        # 两个一样或都不一样
        else:
            return sh_k+2

    def run(self):
        K_num = []
        x = self._input.get_x()
        y = self._input.get_y_raw()
        x = decomposition.PCA(n_components=20).fit_transform(x)
        for _ in range(0, self.repeat_num):
            data_label_folds_ks = np.zeros(
                (y.shape[0], self._cv_repetition, self._k_max - self._k_min + 1)).astype(int)
            sh_y = []
            ch_y = []
            db_y = []
            for i in range(self._cv_repetition):
                for j in self._k_range_list:
                    if self._verbose:
                        print('Applying pyHRDRA for finding %d clusters. Repetition: %d / %d...\n' %
                              (j, i+1, self._cv_repetition))
                    training_final_prediction = hydra_solver_svm(i, x[self._split_index[i][0]], y[self._split_index[i][0]], j, self._output_dir,
                                                                 self._num_consensus, self._num_iteration, self._tol, self._balanced, self._predefined_c,
                                                                 self._weight_initialization_type, self._n_threads, self._save_models, self._verbose)

                    # change the final prediction's label: test data to be 0, the rest training data will b e updated by the model's prediction
                    data_label_fold = y.copy()
                    # all test data to be 0
                    data_label_fold[self._split_index[i][1]] = 0
                    # assign the training prediction
                    data_label_fold[self._split_index[i]
                                    [0]] = training_final_prediction
                    data_label_folds_ks[:, i, j -
                                        self._k_min] = data_label_fold

            x_range = np.arange(self._k_min, self._k_max+1, dtype=int)
            for m in range(self._k_max - self._k_min + 1):
                result = data_label_folds_ks[:, :, m]
                try:
                    sh_y.append(silhouette_score(x, result[:, 0]))
                except:
                    sh_y.append(0)
                try:
                    ch_y.append(calinski_harabasz_score(x, result[:, 0]))
                except:
                    ch_y.append(0)
                try:
                    db_y.append(davies_bouldin_score(x, result[:, 0]))
                except:
                    db_y.append(0)

            sh_y = self.normalization(sh_y)
            ch_y = self.normalization(ch_y)
            db_y = self.normalization(db_y)
            k = self.get_K_from_three_score(sh_y, ch_y, db_y)
            K_num.append(k)


            time_bar(_, self.repeat_num)

        return K_num

       
