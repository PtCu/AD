

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
synthetic_data2 = cwd_path+"/data/feature.tsv"
simulated_cov = cwd_path+"/data/feature.tsv"


def test_hierachical():
    name = "Hierachical"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               HierachicalClusterer, name)


def test_K_medians():
    name = "Kmedians"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               KMedianClusterer, name)


def test_chimera():
    name = "CHIMERA"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               CHIMERAClusterer, name, pt_only=True)


def test_hydra():
    name = "HYDRA"
    print("test "+name)
    # X = {}
    # X["feature"] = simulated_feature
    # X["cov"]=simulated_cov
    # X["outputdir"] = cwd_path+"/"+name+"/output/"
    # utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
    #            HYDRAClusterer, name)
    # true_label = np.append(np.zeros(250), np.ones(250))

    HYDRAClustering(synthetic_data2, cwd_path+"/"+name+"/output/",
                    2, 10, 2, covariate_tsv=None)


def test_lda():
    name = "LDA"
    print("test "+name)
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               LDAClusterer, name)


def test_nmf():
    name = "NMF"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               NMFClusterer, name)


def test_urf():
    name = "URF"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               URFClusterer, name)


def test_Bayesian():
    name = "Bayesian"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID

    utl.eval_K(X, 0.1, 0.5, cwd_path+"/"+name+"/output/"+name,
               BayesianClusterer, name, stride=0.1, get_k_num=True)

    return


def test_louvain():
    name = "Louvain"
    print("test "+name)
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1, decimals=3)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID

    utl.eval_K(X, 0.8, 0.88, cwd_path+"/"+name+"/output/"+name,
               LouvainClusterer, name, stride=0.005, get_k_num=True)


def test_moe():
    name = "MOE"
    print("test "+name)

    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        synthetic_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
               MOEClusterer, name)


if __name__ == "__main__":
    # test_moe()
    # test_chimera()
    test_nmf()
    test_lda()
    # test_urf()
    # test_hierachical()
    # test_K_medians()
    # test_louvain()
    #test_hydra()
    # test_Bayesian()
