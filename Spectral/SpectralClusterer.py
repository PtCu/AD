from pdb import main

import os
import numpy as np

import networkx as nx

from Spectral.main import euler_distance
from community import community_louvain
import itertools
import math
from typing import Counter
import URF.forest_cluster as rfc
from sklearn import manifold,decomposition,cluster
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
        
        # G = nx.Graph()
        # G.add_nodes_from(ID)

        # edges = list(itertools.permutations(ID, 2))
        # for edge in edges:
        #     ROI1 = x_t[edge[0]-1]
        #     ROI2 = x_t[edge[1]-1]
        #     distance = euler_distance(ROI1, ROI2)
        #     G.add_edge(edge[0], edge[1],  weight=distance)

        # # 返回的partition为每个节点最终归属的subtype
        # # resolution表示了一个社区大小。1表示为所有节点划分为一个社区，某些数（大于1）表示每个节点一个社区
        # # 调整resolution，以使其最终能划分为两个社区
        # # 最终的partition为一个字典，key为ID，value为所属社区号
        # partition = community_louvain.best_partition(G, resolution=self.alpha)
        # self.labels_=np.zeros(len(self.x_data))
        # for i,value in partition.items():
        #     self.labels_[i-1]=value
            
        # label_cnt=Counter(self.labels_)
        # self.k_num=len(label_cnt)


    def euler_distance(self,point1, point2):
        """
        计算两点之间的欧拉距离，支持多维
        """
        distance = 0.0
        for a, b in zip(point1, point2):
            distance += math.pow(a - b, 2)
        return math.sqrt(distance)


