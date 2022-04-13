# -*- coding: utf-8 -*-

# License: GPL 3.0
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
import networkx as nx

from Bayesian.bhc import (BayesianHierarchicalClustering,
                 BayesianRoseTrees,
                 NormalInverseWishart)

cwd_path = os.getcwd()

output_file = cwd_path + "/Bayesian-clustering/output/output.tsv"

origin_data = cwd_path + "/Bayesian-clustering/data/data.csv"

simulated_data = cwd_path + "/Bayesian-clustering/data/simulated_data.tsv"

outcome_file = cwd_path + "/Bayesian-clustering/output/oucome.txt"

output_dir = cwd_path + "/Bayesian-clustering/output/"


def read_data(filename):
    sys.stdout.write('\treading data...\n')
    feat_cov = None
    ID = None
    with open(filename) as f:
        data = list(csv.reader(f, delimiter='\t'))
        header = np.asarray(data[0])
        if 'GROUP' not in header:
            sys.stdout.write(
                'Error: group information not found. Please check csv header line for field "Group".\n')
            sys.exit(1)
        if 'ROI' not in header:
            sys.stdout.write(
                'Error: image features not found. Please check csv header line for field "IMG".\n')
            sys.exit(1)
        data = np.asarray(data[1:])

        group = (data[:, np.nonzero(header == 'GROUP')
                 [0]].flatten()).astype(int)
        feat_img = (data[:, np.nonzero(header == 'ROI')[0]]).astype(np.float)
        if 'COVAR' in header:
            feat_cov = (data[:, np.nonzero(header == 'COVAR')[0]]).astype(
                np.float)
        if 'ID' in header:
            ID = data[:, np.nonzero(header == 'ID')[0]]
            ID = ID[group == 1]

    return feat_cov, feat_img, ID, group


def get_data(filename):
    feat_cov, feat_img, ID, group = read_data(filename)

    feat_all = np.hstack((feat_cov, feat_img))

    feat_img = np.transpose(feat_img)

    x_img = feat_img[:, group == 1]  # patients
    x_img = np.transpose(x_img)

    feat_all = np.transpose(feat_all)
    x_all = feat_all[:, group == 1]  # patients
    x_all = np.transpose(x_all)

    return x_img, x_all, ID


def run_bhc(data,alpha=1):
    # Hyper-parameters (these values must be optimized!)
    g = 20
    scale_factor = 0.001

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
    #簇的id
    id = 0
    #将对应连通分量上的叶节点归到相应的簇上
    for c in nx.weakly_connected_components(dag):
        nodeSet = dag.subgraph(c).nodes()
        id = id + 1
        for node in nodeSet:
            if dag.in_degree(node) == 1 and dag.out_degree(node) == 0:
                output_dict[node] = id
    output_label = np.zeros(len(output_dict), dtype=int)
    for key in output_dict:
        output_label[key] = output_dict[key]
    return output_label


def write_outputfile(output_file, ID, label, true_label, outcome_file, name):
    with open(output_file, 'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label)):
                f.write('%d\n' % (label[i] + 1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label)):
                f.write('%s,%d\n' % (ID[i][0], label[i] + 1))

    with open(output_file) as f:
        out_label = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(numpy.int)

    measure = ARI(true_label, out_label)

    with open(outcome_file, 'a') as f:
        f.write(name + " :\n")
        f.write("ARI: " + str(measure) + '\n')


if __name__ == "__main__":
    x_img, x_all, ID = get_data(simulated_data)

    label_bhc = run_bhc(x_img)
    # label_brt = run_brt(x_img)
    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    write_outputfile(output_file, ID, label_bhc,
                     true_label, outcome_file, "bhc")

    # write_outputfile(output_file, ID, label_brt,
    #                  true_label, outcome_file, "brt")
