# -*- coding: utf-8 -*-

# License: GPL 3.0

import sys,os

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from scipy.cluster.hierarchy import dendrogram, linkage

from bhc import (BayesianHierarchicalClustering,
                 BayesianRoseTrees,
                 NormalInverseWishart)

cwd_path = os.getcwd()

output_file = cwd_path + "/Bayesian-clustering/output/output.tsv"

origin_data = cwd_path + "/Bayesian-clustering/data/data.csv"

simulated_data = cwd_path + "/Bayesian-clustering/data/simulated_data.csv"

outcome_file = cwd_path + "/Bayesian-clustering/output/oucome.txt"

output_dir = cwd_path + "/Bayesian-clustering/output/"

def main():
    data = np.genfromtxt(origin_data, delimiter=',')

    plot_data(data)

    run_linkage(data, 'single')
    run_linkage(data, 'complete')
    run_linkage(data, 'average')

    run_bhc(data)
    run_brt(data)


def plot_data(data):
    plt.style.use('seaborn-poster')
    plt.figure(facecolor="white", figsize=(6, 4))
    ax = plt.gca()
    ax.set_axisbelow(True)

    ax.grid(True, color='lightgrey', linestyle='-', alpha=0.4)
    ax.tick_params(axis='both', which='both', length=0, labelcolor='0.5')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for i in range(data.shape[0]):
        if i < 10:
            plt.scatter(data[i, 0], data[i, 1], s=200, color='#1F77B4',
                        marker=r"$ {} $".format(str(i)))
        else:
            plt.scatter(data[i, 0], data[i, 1], s=200, color='#FF7F0E',
                        marker=r"$ {} $".format(str(i)))

    plt.xlabel('x', fontsize=18, weight='light', color='0.35')
    plt.ylabel('y', fontsize=18, weight='light', color='0.35')

    plt.xticks(np.arange(0, 8 + 1, 1), fontsize=14)
    plt.yticks(np.arange(0, 11 + 1, 1), fontsize=14)

    plt.draw()
    plt.savefig(os.path.join(output_dir,'data_plot.png'), format='png', dpi=100)


def run_linkage(data, method):
    plt.clf()
    Z = linkage(data, method)
    dendrogram(Z)
    plt.draw()
    plt.savefig(
        os.path.join(output_dir, 'linkage_{0}_plot.png'.format(method)), format='png', dpi=100)


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

    build_graph(bhc_result.node_ids,
                bhc_result.arc_list,
                os.path.join(output_dir, 'bhc_plot'))


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

    build_graph(brt_result.node_ids,
                brt_result.arc_list,
                os.path.join(output_dir, 'brt_plot'))


def build_graph(node_ids, arc_list, filename):
    dag = Digraph()

    for id in node_ids:
        dag.node(str(id))

    for arc in arc_list:
        dag.edge(str(arc.source), str(arc.target))

    dag.render(filename=filename, format='png', cleanup=True)


if __name__ == "__main__":
    main()
    sys.exit()
