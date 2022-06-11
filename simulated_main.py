
# 仿真数据1

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

sys.path.append(os.getcwd())
cwd_path = os.getcwd()


simulated_data1 = cwd_path+"/data/simulated_data1.tsv"
simulated_data2 = cwd_path+"/data/simulated_data2.tsv"

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
    return utl.get_final_K(X, K_min, K_max, CHIMERAClusterer)


def test_hydra(data_file, label):
    name = "HYDRA"
    print("test "+name)

    HYDRAClustering(data_file, cwd_path+"/"+name+"/output/",
                    K_min, K_max, 10, label=label, covariate_tsv=None)


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
    name = "Spectral"
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


def test_all(simulated1, simulated2,label):
    # 得到若干次评估的最优聚类数目
    nmf = test_nmf(simulated1, label)
    gmm = test_gmm(simulated1, label)
    lda = test_lda(simulated1, label)
    urf = test_urf(simulated1, label)
    hierachical = test_hierachical(simulated1, label)
    k_medians = test_K_medians(simulated1, label)
    spectral = test_spectral(simulated1, label)
    chimera = test_chimera(simulated1, label)
    hydra = test_hydra(simulated2, label)
    
    with open(output_dir+label+".txt", 'a') as f:
        f.write("gmm: "+str(gmm)+"\n")
        f.write("nmf: "+str(nmf)+"\n")
        f.write("lda: "+str(lda)+"\n")
        f.write("urf: "+str(urf)+"\n")
        f.write("hierachical: "+str(hierachical)+"\n")
        f.write("k_medians: "+str(k_medians)+"\n")
        f.write("spectral: "+str(spectral)+"\n")
        f.write("chimera: "+str(chimera)+"\n")
        f.write("chimera: "+str(hydra)+"\n")
        f.close()

    # 绘制箱线图
    data = {}
    data["nmf"] = nmf
    data["gmm"] = gmm
    data["lda"] = lda
    data["urf"] = urf
    data["hierachical"] = hierachical
    data["k_medians"] = k_medians
    data["spectral"] = spectral
    data["chimera"] = chimera
    data["hydra"] = hydra

    labels, data = data.keys(), data.values()
    plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

    plt.figure(figsize=(12,7))
    plt.boxplot(data)
    plt.ylabel("K",fontsize=20)
    plt.xlabel("Algorithms",fontsize=20)  # 我们设置横纵坐标的标题。
    plt.xticks(range(1, len(labels) + 1), labels,fontsize=15)
    plt.yticks(range(1,9),fontsize=16)
    #plt.show()
    plt.savefig(output_dir+label)


if __name__ == "__main__":
    test_all(simulated_data1, simulated_data2,"simulated")
