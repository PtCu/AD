from scipy import rand
from sklearn import cluster, manifold
from pdb import main

import os
import sys
import csv
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
import forest_cluster as rfc
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/URF/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/URF/output/output_without_cov.tsv"

simulated_data = cwd_path+"/URF/data/simulated_data.tsv"

outcome_file = cwd_path+"/URF/output/oucome.txt"

output_dir = cwd_path + "/URF/output/"

K = 2


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


def get_data(filename):
    feat_cov, feat_img, ID, group = read_data(filename)

    feat_all = np.hstack((feat_cov, feat_img))

    x_img = feat_img[group == 1, :]

    x_all = feat_all[group == 1, :]

    return x_img, x_all, feat_img, feat_all, ID


def clustering(X, k, step=50):
    """
    Random Forest clustering works as follows
    1. Construct a dissimilarity measure using RF
    2. Use an embedding algorithm (MDS, TSNE) to embed into a 2D space preserving that dissimilarity measure.
    3. Cluster using K-means or K-medoids
    """
    rf = rfc.RandomForestEmbedding(
        n_estimators=step, random_state=10, n_jobs=-1, sparse_output=False)
    # 其中，leaves[i][j]表示数据i经过n_estimator个决策树后，
    # 决策树j最终的做出的决策。每个决策树最终做出的决策为该决策树的叶节点编号。
    leaves = rf.fit_transform(X)
    projector = manifold.TSNE(
        n_components=2, random_state=1234, metric='hamming')
    embedding = projector.fit_transform(leaves)
    clusterer = cluster.KMeans(n_clusters=k, random_state=1234, n_init=20)
    clusterer.fit(embedding)
    label = clusterer.labels_
    return label


def eval_K(X, k_min, k_max, filename):
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        label = clustering(X, k)
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("URF")
    plt.xlabel("K range")
    plt.ylabel("Silhoutte score")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+filename)
    plt.clf()


if __name__ == "__main__":

    x_img, x_all, feat_img, feat_all, ID = get_data(simulated_data)

    eval_K(feat_img, 2, 10, "PT_NC.png")
    # eval_K(x_img, 2, 10, "only_PT.png")

    # step = 500

    # label2 = clustering(x_img, step)

    # true_label = np.append(np.zeros(250), np.ones(250))

    # write_outputfile(output_file_without_cov, ID, label2,
    #                  true_label, outcome_file, "N= "+str(step)+" without covariate")
