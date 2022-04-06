from collections import Counter
from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from sklearn import cluster
import random
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
cwd_path = os.getcwd()


class NMFClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []
        self.x_data =[]

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        source_flatten = self.x_data.flatten()
        count_data = Counter(source_flatten)
        Mx = np.zeros([len(self.x_data), len(count_data)], dtype=int)
        for i in range(len(self.x_data)):
            for j in range(len(self.x_data[0])):
                Mx[i][j] = count_data[self.x_data[i][j]]
                
        model = NMF(n_components=self.cluster_num)
        W = model.fit_transform(Mx)
        H = model.components_
        label = np.zeros(len(self.x_data), dtype=int)
        judge = W
        for i in range(len(judge)):
            subtype = 0
            max_prob = 0
            for j in range(self.cluster_num):
                if(judge[i][j] > max_prob):
                    max_prob = judge[i][j]
                    subtype = j
            label[i] = subtype
        self.labels_=label

