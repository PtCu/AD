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
import HYDRA.mlni.clustering as hydra

cwd_path = os.getcwd()


class HYDRAClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.labels_ = []

    def fit(self, X):
        self.labels_ = hydra.clustering(self.cluster_num, X)
