

from operator import imod
from unicodedata import decimal

from sklearn.decomposition import LatentDirichletAllocation
from Louvain.LouvainClusterer import LouvainClusterer
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
from MOE.MOEClusterer import MOEClusterer
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

sys.path.append(os.getcwd())
cwd_path = os.getcwd()

synthetic_data1 = cwd_path+"/data/synthetic_data1.csv"
synthetic_data2 = cwd_path+"/data/synthetic_data2.csv"
simulated_data2 = cwd_path+"/data/feature.tsv"
simulated_data1 = cwd_path+"/data/simulated_data1.tsv"
real_data1 = cwd_path+"/data/real_data1.csv"
real_data2 = cwd_path+"/data/real_data2.csv"


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

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               HierachicalClusterer, title=label)


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

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               KMedianClusterer, title=label)


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

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               CHIMERAClusterer, title=label)


def test_hydra(data_file, label):
    name = "HYDRA"
    print("test "+name)
    # X = {}
    # X["feature"] = simulated_feature
    # X["cov"]=simulated_cov
    # X["outputdir"] = cwd_path+"/"+name+"/output/"
    # utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
    #            HYDRAClusterer, name)
    # true_label = np.append(np.zeros(250), np.ones(250))

    HYDRAClustering(data_file, cwd_path+"/"+name+"/output/",
                    2, 10, 2, label=label, covariate_tsv=None)


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
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               LDAClusterer, label)


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

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               NMFClusterer, label)


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

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               URFClusterer, label)


def test_Bayesian(data_file, label):
    name = "Bayesian"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID

    utl.eval_K(X, 0.1, 0.5, cwd_path+"/"+name+"/output/"+name,
               BayesianClusterer, label, stride=0.1, get_k_num=True)


def test_louvain(data_file, label):
    name = "Louvain"
    print("test "+name)
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file, decimals=3)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID

    utl.eval_K(X, 0.8, 0.88, cwd_path+"/"+name+"/output/"+name,
               LouvainClusterer, label, stride=0.005, get_k_num=True)


def test_moe(data_file, label):
    name = "MOE"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        data_file)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               MOEClusterer, label)


if __name__ == "__main__":
    # test_moe(synthetic_data1,"synthetic")

    # test_nmf(synthetic_data1,"synthetic")
    # test_lda(synthetic_data1,"synthetic")
    # test_urf(synthetic_data1,"synthetic")
    # test_hierachical(synthetic_data1,"synthetic")
    # test_K_medians(synthetic_data1,"synthetic")
    test_louvain(synthetic_data1, "synthetic")

    # test_moe(simulated_data1,"simulated")

    # test_nmf(simulated_data1,"simulated")
    # test_lda(simulated_data1,"simulated")
    # test_urf(simulated_data1,"simulated")
    # test_hierachical(simulated_data1,"simulated")
    # test_K_medians(simulated_data1,"simulated")
    test_louvain(simulated_data1, "simulated")
    # test_chimera(simulated_data1,"simulated")
    # test_chimera(synthetic_data1,"synthetic")
    test_hydra(synthetic_data2, "synthetic")
    test_hydra(simulated_data2, "simulated")

    test_moe(real_data1, "real")

    test_nmf(real_data1, "real")
    test_lda(real_data1, "real")
    test_urf(real_data1, "real")
    test_hierachical(real_data1, "real")
    test_K_medians(real_data1, "real")
    test_louvain(real_data1, "real")
    test_chimera(real_data1, "real")
    test_chimera(real_data1, "real")
    test_hydra(real_data2, "real")
