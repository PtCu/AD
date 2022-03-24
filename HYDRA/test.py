from mlni.hydra_clustering import clustering
from dis import dis
import itertools
from community import community_louvain, induced_graph
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn import cluster
from typing import Counter

cwd_path = os.getcwd()

covariate_tsv = cwd_path+"/HYDRA/data/covariate.tsv"

feature_tsv = cwd_path+"/HYDRA/data/feature.tsv"

output_dir = cwd_path + "/HYDRA/output"

k_min = 2
k_max = 2
cv_repetition = 2

clustering(feature_tsv, output_dir, k_min, k_max,
           cv_repetition, covariate_tsv=covariate_tsv)
