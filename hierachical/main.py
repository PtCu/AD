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

if __name__ == "__main__":

    # ================================= Reading Data ======================================================
    sys.stdout.write('\treading data...\n')
    feat_cov = None
    feat_set = None
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

    feat_all = np.hstack((feat_cov, feat_img))

    feat_img = np.transpose(feat_img)
    x_img = feat_img[:, group == 1]  # patients
    x_img = np.transpose(x_img)

    sk = cluster.AgglomerativeClustering(K)

    feat_all = np.transpose(feat_all)
    x_all = feat_all[:, group == 1]  # patients
    x_all = np.transpose(x_all)
    # Without covariate
    sk.fit(x_img)
    label_without_cov = sk.labels_

    with open(output_file_with_cov, 'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label_without_cov)):
                f.write('%d\n' % (label_without_cov[i]+1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label_without_cov)):
                f.write('%s,%d\n' % (ID[i][0], label_without_cov[i]+1))

    # With covariate
    sk.fit(x_all)
    label_with_cov = sk.labels_
    with open(output_file_without_cov, 'w') as f:
        if ID is None:
            f.write('Cluster\n')
            for i in range(len(label_with_cov)):
                f.write('%d\n' % (label_with_cov[i]+1))
        else:
            f.write('ID,Cluster\n')
            for i in range(len(label_with_cov)):
                f.write('%s,%d\n' % (ID[i][0], label_with_cov[i]+1))

    with open(output_file_with_cov) as f:
        out_label1 = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label1[0] == "Cluster")[0]
    out_label1 = out_label1[1:, idx].flatten().astype(numpy.int)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    measure = ARI(true_label, out_label1)

    with open(outcome_file, 'a') as f:
        f.write("Hierachical with covariate:\n")
        f.write("ARI: " + str(measure)+'\n')

    with open(output_file_without_cov) as f:
        out_label2 = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label2[0] == "Cluster")[0]
    out_label2 = out_label2[1:, idx].flatten().astype(numpy.int)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    measure = ARI(true_label, out_label2)

    with open(outcome_file, 'a') as f:
        f.write("Hierachical without covariate:\n")
        f.write("ARI: " + str(measure)+'\n')
