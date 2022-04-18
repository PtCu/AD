

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

simulated_data1 = cwd_path+"/data/simulated_data1.tsv"
simulated_feature = cwd_path+"/data/covariate.tsv"
simulated_cov = cwd_path+"/data/feature.tsv"


def test_hierachical():
    name = "Hierachical"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov,_, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               HierachicalClusterer, name,get_k_num=True)


def test_K_medians():
    name = "Kmedians"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov,_, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               KMedianClusterer, name)


def test_chimera():
    name = "CHIMERA"
    print("test "+name)
    true_label = np.append(np.ones(250), np.ones(250)*2)

    pt_nc_img, pt_nc_cov,_, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 500
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
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
    true_label = np.append(np.zeros(250), np.ones(250))

    HYDRAClustering(simulated_feature, cwd_path+"/"+name+"/output/",
                    2, 10, 2, true_label, covariate_tsv=simulated_cov)


def test_lda():
    name = "LDA"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov,_, ID, group = utl.get_data(
        simulated_data1, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               LDAClusterer, name)


def test_nmf():
    name = "NMF"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov,_, ID, group = utl.get_data(
        simulated_data1, decimals=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               NMFClusterer, name)


def test_urf():
    name = "URF"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               URFClusterer, name)
    


def test_Bayesian():
    name = "Bayesian"
    print("test "+name)
    # true_label = np.append(
    # #     np.zeros(500), np.append(np.ones(250), np.ones(250)*2))
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))
    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 0.1, 0.5, cwd_path+"/"+name+"/output/"+name+".png",
               BayesianClusterer, name,stride=0.1,get_k_num=True)

    return


def test_louvain():
    name = "Louvain"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))
    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1, decimals=3)
    X = {}
    X["group"] = group
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 0.8, 0.88, cwd_path+"/"+name+"/output/"+name+".png",
               LouvainClusterer, name,stride=0.005,get_k_num=True)



def test_moe():
    name = "MOE"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    X["true_label"] = true_label
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               MOEClusterer, name)
 


if __name__ == "__main__":
    # test_moe()
    # test_hydra()
    # test_chimera()
    # test_nmf()
    # test_lda()
    # test_urf()
    # test_hierachical()
    # test_K_medians()
    test_louvain()
    #test_Bayesian()
    
    
    
    
