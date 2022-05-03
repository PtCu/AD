from pdb import main

import os
import HYDRA.mlni.hydra_clustering as hydra

cwd_path = os.getcwd()


class HYDRAClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []
        self.x_data=[]
        self.y_data=[]

    def fit(self, X):
        self.labels_,self.x_data,self.y_data = hydra.clustering_one_round(X["feature"],X["outputdir"],self.cluster_num)
