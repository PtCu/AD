from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn import cluster

cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/hierachical/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/hierachical/output/output_without_cov.tsv"

test_data = cwd_path+"/hierachical/data/simulated_data.tsv"

outcome_file = cwd_path+"/hierachical/output/oucome.txt"

K = 2


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


if __name__ == "__main__":

    feat_cov, feat_img, ID, group = read_data()

    feat_all = np.hstack((feat_cov, feat_img))

    feat_img = np.transpose(feat_img)

    x_img = feat_img[:, group == 1]  # patients
    x_img = np.transpose(x_img)

    sk = cluster.AgglomerativeClustering(K)

    feat_all = np.transpose(feat_all)
    x_all = feat_all[:, group == 1]  # patients
    x_all = np.transpose(x_all)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)
    # With covariate
    sk.fit(x_all)
    label_with_cov = sk.labels_
    write_outputfile(output_file_with_cov, ID, label_with_cov,
                     true_label, outcome_file, "Hierachical with covariate")

    # Without covariate
    sk.fit(x_img)
    label_without_cov = sk.labels_
    write_outputfile(output_file_without_cov, ID, label_without_cov,
                     true_label, outcome_file, "Hierachical without covariate")
