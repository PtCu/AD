from pdb import main

import os
from statistics import mode
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


class HierachicalClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = int(cluster_num)
        self.labels_ = []
        self.x_data = []
        self.y_data=[]
        self.name="Hierachical"

    def fit(self, X):
        self.x_data = X["pt_nc_img"]
        model = cluster.AgglomerativeClustering(self.cluster_num)
        model.fit(self.x_data)
        self.labels_=model.labels_
        #self.y_data=X["true_label"]
