from scipy import rand
from sklearn import cluster, manifold
from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
import forest_cluster as rfc
import numpy as np
from sklearn import cluster
from sklearn.ensemble import RandomTreesEmbedding
from sklearn import manifold

cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/URF/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/URF/output/output_without_cov.tsv"

simulated_data = cwd_path+"/URF/data/simulated_data.tsv"

outcome_file = cwd_path+"/URF/output/oucome.txt"

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
        out_label = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(numpy.int)

    measure = ARI(true_label, out_label)

    with open(outcome_file, 'a') as f:
        f.write(name+" :\n")
        f.write("ARI: " + str(measure)+'\n')


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


def clustering(X):
    """
    Random Forest clustering works as follows
    1. Construct a dissimilarity measure using RF
    2. Use an embedding algorithm (MDS, TSNE) to embed into a 2D space preserving that dissimilarity measure.
    3. Cluster using K-means or K-medoids
    """
    rf = rfc.RandomForestEmbedding(
        n_estimators=5000, random_state=10, n_jobs=-1, sparse_output=False)
    leaves = rf.fit_transform(X)
    projector = manifold.TSNE(
        n_components=2, random_state=1234, metric='hamming')
    embedding = projector.fit_transform(leaves)
    clusterer = cluster.KMeans(n_clusters=K, random_state=1234, n_init=20)
    clusterer.fit(embedding)
    label = clusterer.labels_
    return label


if __name__ == "__main__":

    x_img, x_all, ID = get_data(simulated_data)

    label1 = clustering(x_all)
    label2 = clustering(x_img)

    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))
    # With covariate

    write_outputfile(output_file_with_cov, ID, label1,
                     true_label, outcome_file, "Hierachical with covariate")

    # Without covariate

    write_outputfile(output_file_without_cov, ID, label2,
                     true_label, outcome_file, "Hierachical without covariate")
