from pdb import main
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from smt.applications import MOE
from smt.problems import LpNorm
from sklearn import manifold
import forest_cluster as rfc
cwd_path = os.getcwd()

output_file = cwd_path+"/MOE/output/output.tsv"

simulated_data = cwd_path+"/MOE/data/simulated_data.tsv"

outcome_file = cwd_path+"/MOE/output/oucome.txt"

K = 2

PT_NUMS = 500
ROI_NUMS = 20


def stable_sigmoid(x):

    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


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


def clustering(X, k):
    rf = rfc.RandomForestEmbedding(
        n_estimators=5000, random_state=10, n_jobs=-1, sparse_output=False)
    leaves = rf.fit_transform(X)
    projector = manifold.TSNE(
        n_components=2, random_state=1234, metric='hamming')
    x_t = projector.fit_transform(leaves)
    # prob = LpNorm(ndim=2)
    # y_t = prob(x_t)
    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))
    y_t=true_label
    moe = MOE(smooth_recombination=True, n_clusters=k)
    moe.set_training_values(x_t, y_t)
    moe.train()
    A = moe._proba_cluster(x_t)
    label = np.zeros(PT_NUMS, dtype=int)

    for i in range(len(A)):
        subtype = 0
        max_prob = 0
        for j in range(k):
            if(A[i][j] > max_prob):
                max_prob = A[i][j]
                subtype = j
        label[i] = subtype
    return label


if __name__ == "__main__":

    x_img, x_all, ID = get_data(simulated_data)

    out_label = clustering(x_img, K)

    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    write_outputfile(output_file, ID, out_label,
                     true_label, outcome_file, "MOE")
