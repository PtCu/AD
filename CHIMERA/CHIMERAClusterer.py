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
import CHIMERA.chimera.chimera_clustering as chm

cwd_path = os.getcwd()


class CHIMERAClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []
        self.x_data=[]
        self.y_data=[]

    def fit(self, X):
        self.labels_ = []
        self.x_data=[]
        group=X["group"].flatten()
        self.x_data=X["pt_nc_img"][group!=0]
        self.labels_ = chm.clustering(self.cluster_num, X)
        self.y_data = X["true_label"]
