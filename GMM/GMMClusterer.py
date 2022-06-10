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
        self.name="GMM"
        
    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        x_t=decomposition.PCA(n_components=20).fit_transform(self.x_data)
        gmm = GMM(n_components=self.cluster_num).fit(x_t)
        label = gmm.predict(x_t)
        self.labels_=label
