
# 仿真数据2

from operator import imod
from unicodedata import decimal

from sklearn.decomposition import LatentDirichletAllocation
from Spectral.SpectralClusterer import SpectralClusterer
from CHIMERA.CHIMERAClusterer import CHIMERAClusterer
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
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())
cwd_path = os.getcwd()


output_dir = cwd_path+"/output/"

K_min = 2
K_max = 9


def test_hierachical(data_file, label):
    name = "Hierachical"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
               HierachicalClusterer, title=label)
    return utl.get_final_K(X, K_min, K_max, HierachicalClusterer)


def test_K_medians(data_file, label):
    name = "Kmedians"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               KMedianClusterer, title=label)
    return utl.get_final_K(X, K_min, K_max, KMedianClusterer)


def test_chimera(data_file, label):
    name = "CHIMERA"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               CHIMERAClusterer, title=label)
    return utl.get_final_K(X, K_min+1, K_max, CHIMERAClusterer)


def test_hydra(data_file, label):
    name = "HYDRA"
    print("test "+name)

    return HYDRAClustering(data_file, cwd_path+"/"+name+"/output/",
                           K_min, K_max, 1, label=label, covariate_tsv=None)


def test_lda(data_file, label):
    name = "LDA"
    print("test "+name)
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
               LDAClusterer, label)
    return utl.get_final_K(X, K_min, K_max, LDAClusterer)


def test_nmf(data_file, label):
    name = "NMF"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
                NMFClusterer, label)
    return utl.get_final_K(X, K_min, K_max, NMFClusterer)


def test_urf(data_file, label):
    name = "URF"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
               URFClusterer, label)
    return utl.get_final_K(X, K_min, K_max, URFClusterer)


def test_spectral(data_file, label):
    name = "Louvain"
    print("test "+name)
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file, decimals=3)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
       SpectralClusterer, label)
    return utl.get_final_K(X, K_min, K_max, SpectralClusterer)


def test_gmm(data_file, label):
    name = "GMM"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.plot_K(X, K_min, K_max, cwd_path+"/"+name+"/output/"+name,
               GMMClusterer, label)
    return utl.get_final_K(X, K_min, K_max, GMMClusterer)


def test_all(filename1, filename2, label):
    print("running "+filename1)
    hydra = test_hydra(filename2, label)
    nmf = test_nmf(filename1, label)
    gmm = test_gmm(filename1, label)
    lda = test_lda(filename1, label)
    urf = test_urf(filename1, label)
    hierachical = test_hierachical(filename1, label)
    k_medians = test_K_medians(filename1, label)
    spectral = test_spectral(filename1, label)
    chimera = test_chimera(filename1, label)

    with open(output_dir+label+".txt", 'a') as f:
        f.write("label: "+label)
        f.write("gmm: "+str(gmm)+"\n")
        f.write("nmf: "+str(nmf)+"\n")
        f.write("lda: "+str(lda)+"\n")
        f.write("urf: "+str(urf)+"\n")
        f.write("hierachical: "+str(hierachical)+"\n")
        f.write("k_medians: "+str(k_medians)+"\n")
        f.write("spectral: "+str(spectral)+"\n")
        f.write("chimera: "+str(chimera)+"\n")
        f.write("hydra: "+str(hydra)+"\n")
        f.close()

    data = {}
    data["NMF"] = nmf
    data["GMM"] = gmm
    data["LDA"] = lda
    data["URF"] = urf
    data["CHIMERA"] = chimera
    data["层次聚类"] = hierachical
    data["K-Medians"] = k_medians
    data["谱聚类"] = spectral
    data["HYDRA"] = hydra

    labels, data = data.keys(), data.values()
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

    plt.figure(figsize=(12, 7))
    plt.boxplot(data)
    plt.ylabel("K", fontsize=20)
    plt.xlabel("Algorithms", fontsize=20)  # 我们设置横纵坐标的标题。
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=15)
    plt.yticks(range(1, 9), fontsize=16)
    # plt.show()
    plt.savefig(output_dir+label)


prefix = cwd_path+"/data/clustering"
source_file1 = []
source_file2 = []
for k in range(K_min, K_max):
    source_file1.append(prefix+str(k)+"_random_1.csv")
    source_file2.append(prefix+str(k)+"_random_2.csv")
    for t in range(120, 151, 10):
        source_file1.append(prefix+str(k)+"_"+str(t)+"_1.csv")
        source_file2.append(prefix+str(k)+"_"+str(t)+"_2.csv")


if __name__ == "__main__":

    for i in range(len(source_file1)):
        print("K = "+str(i+2))
        test_all(source_file1[i], source_file2[i], "synthetic_"+str(i+2))

