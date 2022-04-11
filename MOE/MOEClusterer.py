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
from smt.applications import MOE
from smt.problems import LpNorm
cwd_path = os.getcwd()


class MOEClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []
        self.x_data = []
        self.y_data = []

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        self.y_data = X["true_label"]
        rf = rfc.RandomForestEmbedding(
            n_estimators=5000, random_state=10, n_jobs=-1, sparse_output=False)
        leaves = rf.fit_transform(self.x_data)
        projector = manifold.TSNE(
            n_components=2, random_state=1234, metric='hamming')
        x_t = projector.fit_transform(leaves)
        prob = LpNorm(ndim=2)
        y_t = prob(x_t)
        # true_label = np.append(np.zeros(250), np.ones(250))
        # y_t = true_label
        moe = MOE(smooth_recombination=True, n_clusters=self.cluster_num)
        moe.set_training_values(x_t, y_t)
        moe.train()
        A = moe._proba_cluster(x_t)
        label = np.zeros(len(self.x_data), dtype=int)

        for i in range(len(A)):
            subtype = 0
            max_prob = 0
            for j in range(self.cluster_num):
                if(A[i][j] > max_prob):
                    max_prob = A[i][j]
                    subtype = j
            label[i] = subtype
        self.labels_=label
