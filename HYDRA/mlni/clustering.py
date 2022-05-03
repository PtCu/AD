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
from utilities.utils import plot_pic
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
        self.repeat_num=5

    def run_one_round(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()
        data_label_folds_ks = np.zeros(
            (y.shape[0], self._k_max - self._k_min + 1)).astype(int)

        for j in self._k_range_list:

            training_final_prediction = hydra_solver_svm(1, x, y, j, self._output_dir,
                                                         self._num_consensus, self._num_iteration, self._tol, self._balanced, self._predefined_c,
                                                         self._weight_initialization_type, self._n_threads, self._save_models, self._verbose)

            data_label_folds_ks[:, j - self._k_min] = training_final_prediction

        print('Estimating clustering stability...\n')
        # for the adjusted rand index, only consider the PT results
        adjusted_rand_index_results = np.zeros(self._k_max - self._k_min + 1)
        index_pt = np.where(y != -1)[0]  # index for PTs
        for m in range(self._k_max - self._k_min + 1):
            # 此时的result保存了多轮训练的结果. result[i]为第i轮训练的结果
            result = data_label_folds_ks[:, m][index_pt]
            adjusted_rand_index_result = cluster_stability(
                result, self._true_label, self._k_range_list[m])
            # saving each k result into the final adjusted_rand_index_results
            adjusted_rand_index_results[m] = adjusted_rand_index_result
            # silhouette = silhouette_score(result)
            # silhouette_y.append(silhouette)
        # silhouette_x = np.arange(self.k_min, self.k_max+1, dtype=int)
        # plt.title("HYDRA")
        # plt.xlabel("K range")
        # plt.ylabel("Silhoutte score")
        # plt.plot(silhouette_x, silhouette_y)
        # # plt.show()
        # plt.savefig(output_dir+"outcome.png")
        # plt.clf()
        print('Computing the final consensus group membership...\n')
        final_assignment_ks = - \
            np.ones(
                (self._input.get_y_raw().shape[0], self._k_max - self._k_min + 1)).astype(int)
        for n in range(self._k_max - self._k_min + 1):
            result = data_label_folds_ks[:, n][index_pt]
            final_assignment_ks_pt = consensus_clustering(
                result, n + self._k_min)
            final_assignment_ks[index_pt, n] = final_assignment_ks_pt + 1
        # 样本被分为-1（NC)以及0（亚型1）和1（亚型2）
        print('Saving the final results...\n')
        # save_cluster_results(adjusted_rand_index_results, final_assignment_ks)
        columns = ['ari_' + str(i) + '_subtypes' for i in self._k_range_list]
        ari_df = pd.DataFrame(
            adjusted_rand_index_results[:, np.newaxis].transpose(), columns=columns)
        ari_df.to_csv(os.path.join(self._output_dir, 'adjusted_rand_index.tsv'), index=False, sep='\t',
                      encoding='utf-8', mode='a')

        # save the final assignment for consensus clustering across different folds
        # df_feature = pd.read_csv(self._feature_tsv, sep='\t')
        # columns = ['assignment_' + str(i) for i in self._k_range_list]
        # participant_df = df_feature.iloc[:, :3]
        # cluster_df = pd.DataFrame(final_assignment_ks, columns=columns)
        # all_df = pd.concat([participant_df, cluster_df], axis=1)
        # all_df.to_csv(os.path.join(self._output_dir, 'clustering_assignment.tsv'), index=False,
        #               sep='\t', encoding='utf-8')

    def cluster_stability(self, X, est, n_iter=20, random_state=None, pt_only=False):
        labels = []
        indices = []
        X_copy = {}
        sample_indices = np.arange(0, X["pt_nc_img"].shape[0])
        for i in range(n_iter):
            # draw bootstrap samples, store indices
            # rng.shuffle(sample_indices)
            sample_indices = rng.randint(
                0, X["pt_nc_img"].shape[0], X["pt_nc_img"].shape[0])
            # X_bootstrap = X[sample_indices]
            X_copy["pt_nc_img"] = X["pt_nc_img"][sample_indices]
            X_copy["pt_nc_cov"] = X["pt_nc_cov"][sample_indices]
            X_copy["pt_ID"] = X["pt_ID"][sample_indices]
            X_copy["group"] = X["group"][sample_indices]
            est.fit(X_copy)
            # sample_indices = rng.randint(
            #     0, X["pt_nc_img"].shape[0], X["pt_nc_img"].shape[0])
            # store clustering outcome using original indices
            if(pt_only):
                tmp_indices = sample_indices[X_copy["group"] != 0]-X["len"]
                indices.append(tmp_indices)
                relabel = -np.ones(X["len"], dtype=np.int)
                relabel[tmp_indices] = est.labels_
            else:
                indices.append(sample_indices)
                relabel = -np.ones(X["pt_nc_img"].shape[0], dtype=np.int)
                relabel[sample_indices] = est.labels_
            # est = clone(est)
            if hasattr(est, "random_state"):
                # randomize estimator if possible
                est.random_state = rng.randint(1e5)

            labels.append(relabel)

        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                # we also compute the diagonal which is a bit silly
                in_both = np.intersect1d(i, j)
                scores.append(ARI(l[in_both], k[in_both]))
        return np.mean(scores)

    def plot_pic(self, title, filename, x_label, y_label, x, y):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, y)
        plt.savefig(filename)
        plt.clf()

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
        for _ in range(0, self.repeat_num):
            x = self._input.get_x()
            y = self._input.get_y_raw()
            data_label_folds_ks = np.zeros(
                (y.shape[0], self._cv_repetition, self._k_max - self._k_min + 1)).astype(int)
            sh_y = []
            ch_y = []
            db_y = []
            x = decomposition.PCA(n_components=20).fit_transform(x)
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
                sh_y.append(silhouette_score(x, result[:, 0]))
                ch_y.append(calinski_harabasz_score(x, result[:, 0]))
                db_y.append(davies_bouldin_score(x, result[:, 0]))
            sh_y = self.normalization(sh_y)
            ch_y = self.normalization(ch_y)
            db_y = self.normalization(db_y)
            k = self.get_K_from_three_score(sh_y, ch_y, db_y)
            K_num.append(k)

            si, = plt.plot(
                x_range, sh_y, label="Silhoutte score", linestyle="solid")
            ar, = plt.plot(x_range, ch_y, label="CH score",
                           linestyle="dashdot")
            st, = plt.plot(x_range, db_y, label="DB score", linestyle="dashed")
            plt.legend([st, si, ar], ["DB score",
                       "Silhoutte score", "CH score"])

            plt.savefig(self._output_dir+"/HYDRA_"+self._label+str(_)+".png")
            plt.clf()
            time_bar(_, self.repeat_num)

        return K_num

        # stability_score=[]
        # for l, i in zip(labels, indices):
        #     for k, j in zip(labels, indices):
        #         # we also compute the diagonal which is a bit silly
        #         in_both = np.intersect1d(i, j)
        #         stability_score.append(
        #             ARI(result[in_both][:,0], result[in_both][:,1]))
        # stability_y.append(np.mean(stability_score))
    # plt.title(title)
        # plt.xlabel("K")
        # plt.ylabel("Score")
        # x_range = np.arange(self._k_min, self._k_max+1, dtype=int)
        # sh_y = self.normalization(sh_y)
        # ch_y = self.normalization(ch_y)
        # db_y = self.normalization(db_y)

        # si, = plt.plot(x_range, sh_y, label="Silhoutte score",linestyle="solid")
        # ar, = plt.plot(x_range, ch_y, label="CH score",linestyle="dashdot")
        # st, = plt.plot(x_range, db_y, label="DB score",linestyle="dashed")
        # plt.legend([st, si, ar], ["DB score", "Silhoutte score", "CH score"])

        # plt.savefig(self._output_dir+"/HYDRA_"+self._label+".png")
        # plt.clf()

        # plot_pic("Silhoutte Score", self._output_dir+"/HYDRA_"+self._label +
        #          "sh.png", "number of clusters", "Silhoutte Score", x_range, silhouette_y)

        # plot_pic("Calinski Harabasz Score", self._output_dir+"/HYDRA_"+self._label +
        #          "ch.png", "number of clusters", "Calinski Harabasz Score", x_range, ch_y)

        # plot_pic("Davies Bouldin Score", self._output_dir+"/HYDRA_"+self._label +
        #          "db.png", "number of clusters", "Davies Bouldin Score", x_range, db_y)

        # print('Computing the final consensus group membership...\n')
        # final_assignment_ks = -np.ones((self._input.get_y_raw().shape[0], self._k_max - self._k_min + 1)).astype(int)
        # for n in range(self._k_max - self._k_min + 1):
        #     result = data_label_folds_ks[:, :, n][index_pt]
        #     final_assignment_ks_pt = consensus_clustering(result, n + self._k_min)
        #     final_assignment_ks[index_pt, n] = final_assignment_ks_pt + 1
        # #样本被分为-1（NC)以及0（亚型1）和1（亚型2）
        # print('Saving the final results...\n')
        # # save_cluster_results(adjusted_rand_index_results, final_assignment_ks)
        # columns = ['ari_' + str(i) + '_subtypes' for i in self._k_range_list]
        # ari_df = pd.DataFrame(adjusted_rand_index_results[:, np.newaxis].transpose(), columns=columns)
        # ari_df.to_csv(os.path.join(self._output_dir, 'adjusted_rand_index.tsv'), index=False, sep='\t',
        #               encoding='utf-8')

        # # save the final assignment for consensus clustering across different folds
        # df_feature = pd.read_csv(self._feature_tsv, sep='\t')
        # columns = ['assignment_' + str(i) for i in self._k_range_list]
        # participant_df = df_feature.iloc[:, :3]
        # cluster_df = pd.DataFrame(final_assignment_ks, columns=columns)
        # all_df = pd.concat([participant_df, cluster_df], axis=1)
        # all_df.to_csv(os.path.join(self._output_dir, 'clustering_assignment.tsv'), index=False,
        #               sep='\t', encoding='utf-8')
