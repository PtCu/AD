
import os

from sklearn import decomposition,cluster
cwd_path = os.getcwd()


class SpectralClusterer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.labels_ = []
        self.x_data = []
        self.y_data = []
        self.k_num = 0
        self.name="Spectral"

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        #self.y_data=X["true_label"]
        x_t=decomposition.PCA(n_components=20).fit_transform(self.x_data)
        y_pred=cluster.SpectralClustering(n_clusters=self.alpha).fit_predict(x_t)
        self.labels_=y_pred
        



