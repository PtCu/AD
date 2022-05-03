from pdb import main

import os



from sklearn import decomposition

from sklearn.mixture import GaussianMixture as GMM
cwd_path = os.getcwd()


class GMMClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = int(cluster_num)
        self.labels_ = []
        self.x_data = []
        self.y_data = []

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        #self.y_data = X["true_label"]
        # rf = rfc.RandomForestEmbedding(
        #     n_estimators=200, random_state=10, n_jobs=-1, sparse_output=False)
        # leaves = rf.fit_transform(self.x_data)
        # projector = manifold.TSNE(
        #     n_components=2, random_state=1234, metric='hamming')
        # x_t = projector.fit_transform(leaves)
        x_t=decomposition.PCA(n_components=20).fit_transform(self.x_data)
        gmm = GMM(n_components=self.cluster_num).fit(x_t)
        label = gmm.predict(x_t)
        # prob = LpNorm(ndim=2)
        # y_t = prob(x_t)
        # true_label = np.append(np.zeros(250), np.ones(250))
        # y_t = true_label
        # moe = MOE(smooth_recombination=False, n_clusters=self.cluster_num)
        # moe.set_training_values(x_t, y_t)
        # moe.train()
        # A = moe._proba_cluster(x_t)
        # label = np.zeros(len(self.x_data), dtype=int)

        # for i in range(len(A)):
        #     subtype = 0
        #     max_prob = 0
        #     for j in range(self.cluster_num):
        #         if(A[i][j] > max_prob):
        #             max_prob = A[i][j]
        #             subtype = j
        #     label[i] = subtype
        self.labels_=label
