# -*- coding: utf-8 -*-

# License: GPL 3.0

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from sklearn.metrics import adjusted_rand_score as ARI
import pandas as pd
from bhc import (BayesianHierarchicalClustering,
                 BayesianRoseTrees,
                 NormalInverseWishart)

cwd_path = os.getcwd()

output_file = cwd_path + "/Bayesian-clustering/output/output.tsv"

origin_data = cwd_path + "/Bayesian-clustering/data/data.csv"

simulated_data = cwd_path + "/Bayesian-clustering/data/simulated_data.csv"

outcome_file = cwd_path + "/Bayesian-clustering/output/oucome.txt"

output_dir = cwd_path + "/Bayesian-clustering/output/"


def run_bhc(data):
    # Hyper-parameters (these values must be optimized!)
    g = 20
    scale_factor = 0.001
    alpha = 1

    model = NormalInverseWishart.create(data, g, scale_factor)

    bhc_result = BayesianHierarchicalClustering(data,
                                                model,
                                                alpha,
                                                cut_allowed=True).build()

    return get_label(bhc_result.node_ids,
                     bhc_result.arc_list)


def run_brt(data):
    # Hyper-parameters (these values must be optimized!)
    g = 10
    scale_factor = 0.001
    alpha = 0.5

    model = NormalInverseWishart.create(data, g, scale_factor)

    brt_result = BayesianRoseTrees(data,
                                   model,
                                   alpha,
                                   cut_allowed=True).build()

    return get_label(brt_result.node_ids,
                     brt_result.arc_list)


def get_label(node_ids, arc_list):
    dag = nx.DiGraph()
    # 转化为图结构
    for id in node_ids:
        dag.add_node(id)

    for arc in arc_list:
        dag.add_edge(arc.source, arc.target)

    output_dict = {}
    i = 0

    for c in nx.weakly_connected_components(dag):
        nodeSet = dag.subgraph(c).nodes()
        i = i + 1
        for node in nodeSet:
            if dag.in_degree(node) == 1 and dag.out_degree(node) == 0:
                output_dict[node] = i
    output_label = np.zeros(len(output_dict), dtype=int)
    for key in output_dict:
        output_label[key] = output_dict[key]
    return output_label


if __name__ == "__main__":
    data = np.genfromtxt(origin_data, delimiter=',')

    label1 = run_bhc(data)
    label2 = run_brt(data)
