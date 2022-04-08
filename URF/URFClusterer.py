from pdb import main

import os
import sys
import csv
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from sklearn import cluster, manifold
import random
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import URF.forest_cluster as rfc
cwd_path = os.getcwd()


class URFClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []
        self.x_data = []
        self.y_data=[]

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        self.y_data = X["true_label"]
        rf = rfc.RandomForestEmbedding(
            n_estimators=50, random_state=10, n_jobs=-1, sparse_output=False)
        # 其中，leaves[i][j]表示数据i经过n_estimator个决策树后，
        # 决策树j最终的做出的决策。每个决策树最终做出的决策为该决策树的叶节点编号。
        leaves = rf.fit_transform(self.x_data)
        projector = manifold.TSNE(
            n_components=2, random_state=1234, metric='hamming')
        embedding = projector.fit_transform(leaves)
        clusterer = cluster.KMeans(n_clusters=self.cluster_num, random_state=1234, n_init=20)
        clusterer.fit(embedding)
        self.labels_ = clusterer.labels_

