
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from operator import imod
from unicodedata import decimal

from sklearn.decomposition import LatentDirichletAllocation
from Spectral.SpectralClusterer import SpectralClusterer
from Bayesian.BayesianClusterer import BayesianClusterer
from CHIMERA.CHIMERAClusterer import CHIMERAClusterer
from HYDRA.HYDRAClusterer import HYDRAClusterer
from HYDRA.mlni.hydra_clustering import clustering as HYDRAClustering
from Kmedians.KMedianClusterer import KMedianClusterer
from Hierachical.HierachicalClusterer import HierachicalClusterer

from pkg_resources import IResourceProvider
from LDA.LDAClusterer import LDAClusterer
from NMF.NMFClusterer import NMFClusterer
from URF.URFClusterer import URFClusterer
from GMM.GMMClusterer import GMMClusterer
import utilities.utils as utl
from ctypes import util
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster
import sys
import pandas as pd
from synthetic_main import test_nmf, test_hierachical

sys.path.append(os.getcwd())
cwd_path = os.getcwd()

synthetic_data1 = cwd_path+"/data/clustering2_1.csv"
synthetic_data2 = cwd_path+"/data/synthetic_data2.csv"
simulated_data2 = cwd_path+"/data/feature.tsv"
simulated_data1 = cwd_path+"/data/simulated_data1.tsv"
real_data1 = cwd_path+"/data/real_data1.csv"
real_data2 = cwd_path+"/data/real_data2.csv"

output_dir = cwd_path+"/output/"

K_min = 2
K_max = 9

#https://www.aiuai.cn/aifarm1335.html

from sklearn.metrics import confusion_matrix,adjusted_rand_score
label1 =[1,0,1,1,1,0,0,1,1]
label2 = [0,1,0,0,0,1,1,0,0]
labels = [0, 1]
cm = confusion_matrix(label1, label2, labels)
ari=adjusted_rand_score(label1,label2)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("demo")
plt.show()



