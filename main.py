

from operator import imod
from unicodedata import decimal

from sklearn.decomposition import LatentDirichletAllocation
from CHIMERA.CHIMERAClusterer import CHIMERAClusterer
# from HYDRA.HYDRAClusterer import HYDRAClusterer
from Kmedians.KMedianClusterer import KMedianClusterer
from Hierachical.HierachicalClusterer import HierachicalClusterer

from pkg_resources import IResourceProvider
from LDA.LDAClusterer import LDAClusterer
from NMF.NMFClusterer import NMFClusterer
from URF.URFClusterer import URFClusterer
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

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               HierachicalClusterer, name, true_label)


def test_K_medians():
    name = "Kmedians"
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
    X["len"]=1000
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               KMedianClusterer, name, true_label)


def test_chimera():
    name = "CHIMERA"
    print("test "+name)
    true_label = np.append(np.ones(250), np.ones(250)*2)

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"]=500
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               CHIMERAClusterer, name, true_label)


def test_hydra():
    name = "HYDRA"
    print("test "+name)
    true_label = np.append(np.ones(250), np.ones(250)*2)

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 500
    # utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
    #            HYDRAClusterer, name, true_label)

    return


def test_lda():
    name = "LDA"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1, decimal=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               LDAClusterer, name, true_label)


def test_nmf():
    name = "NMF"
    print("test "+name)
    true_label = np.append(
        np.zeros(500), np.append(np.ones(250), np.ones(250)*2))

    pt_nc_img, pt_nc_cov, ID, group = utl.get_data(
        simulated_data1, decimal=3)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group
    X["len"] = 1000
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               NMFClusterer, name, true_label)


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
    utl.eval_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name+".png",
               URFClusterer, name, true_label)
    return


def test_Bayesian():
    return


def test_louvain():
    return


def test_sustain():
    return


def test_moe():
    return


if __name__ == "__main__":

    # test_hierachical()
    # test_K_medians()
    test_chimera()
    test_nmf()
    test_lda()
    test_urf()
