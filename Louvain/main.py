# https://resetran.top/community-detection/
# https://resetran.top/community-detection/#toc-heading-29
import itertools
from community import community_louvain
import networkx as nx
import os
import sys
import csv
from sklearn.metrics import adjusted_rand_score as ARI
import math
import numpy as np
from typing import Counter

cwd_path = os.getcwd()

output_file = cwd_path+"/Louvain/output/output.tsv"

test_data = cwd_path+"/Louvain/data/simulated_data.tsv"

outcome_file = cwd_path+"/Louvain/output/oucome.txt"

K = 2

PT_NUMS = 500
ROI_NUMS = 20


def read_data():
    sys.stdout.write('\treading data...\n')
    feat_cov = None
    ID = None
    with open(test_data) as f:
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
            ID = (data[:, np.nonzero(header == 'ID')[0]]).astype(int)
            ID = ID[group == 1]

    return feat_cov, np.around(feat_img, decimals=4), ID, group


def euler_distance(point1, point2):
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

# 用某种方法来度量两个向量的相似度，匹配到最为相似的那个subtype上


def get_subtype(ROIS, subtypes, k):

    ans = 0
    min_dis = float('inf')
    for i in range(k):
        subtype = subtypes[i]
        dis = euler_distance(ROIS, subtype)
        if dis < min_dis:
            min_dis = dis
            ans = i
    return ans


def write_outputfile(output_file, ID, label, true_label, outcome_file, name):
    with open(output_file, 'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label)):
                f.write('%d\n' % (label[i]+1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label)):
                f.write('%s,%d\n' % (ID[i][0], label[i]+1))

    with open(output_file) as f:
        out_label = np.asarray(list(csv.reader(f)))

    idx = np.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(np.int)

    measure = ARI(true_label, out_label)

    with open(outcome_file, 'a') as f:
        f.write(name+" :\n")
        f.write("ARI: " + str(measure)+'\n')


def add_edge(G, ROIS):
    edges = list(itertools.permutations(ROIS, 2))
    G.add_edges_from(edges)
    return


if __name__ == "__main__":

    feat_cov, feat_img, ID, group = read_data()

    feat_img = np.transpose(feat_img)

    x_img = np.transpose(feat_img[:, group == 1])  # patients

    x_img_flatten = x_img.flatten()
    count_data = Counter(x_img_flatten)

    ID_flatten = ID.flatten()

    G = nx.Graph()
    G.add_nodes_from(ID_flatten)

    edges = list(itertools.permutations(ID_flatten, 2))
    for edge in edges:
        ROI1 = x_img[edge[0]-PT_NUMS-1]
        ROI2 = x_img[edge[1]-PT_NUMS-1]
        distance = euler_distance(ROI1, ROI2)
        G.add_edge(edge[0], edge[1],  weight=distance)

    # 返回的partition为每个节点最终归属的subtype
    # resolution表示了一个社区大小。1表示为所有节点划分为一个社区，某些数（大于1）表示每个节点一个社区
    # 调整resolution，以使其最终能划分为两个社区
    # 最终的partition为一个字典，key为ID，value为所属社区号
    partition = community_louvain.best_partition(G, resolution=1.5)

    output_label = []
    for value in partition.values():
        output_label.append(value)

    true_label = np.append(np.ones(250), np.ones(250)*2)

    write_outputfile(output_file, ID, output_label,
                     true_label, outcome_file, "Louvain")
