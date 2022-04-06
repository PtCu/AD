from collections import Counter
from pdb import main

import os
import sys
import csv
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from sklearn import cluster
import random
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import lda
cwd_path = os.getcwd()

class LDAClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_=[]
        self.x_data=[]

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        source_flatten = self.x_data.flatten()
        count_data = Counter(source_flatten)
        Mx = np.zeros([len(self.x_data), len(count_data)], dtype=int)
        for i in range(len(self.x_data)):
            for j in range(len(self.x_data[0])):
                Mx[i][j] = count_data[self.x_data[i][j]]

        model = lda.LDA(n_topics=self.cluster_num, n_iter=150, random_state=1)
        A = model.fit_transform(Mx)  # model.fit_transform(X) is also available

        self.labels_ = np.zeros(len(self.x_data), dtype=int)

        for i in range(len(A)):
            subtype = 0
            max_prob = 0
            for j in range(self.cluster_num):
                if(A[i][j] > max_prob):
                    max_prob = A[i][j]
                    subtype = j
            self.labels_[i] = subtype



