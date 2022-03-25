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
from sklearn.naive_bayes import GaussianNB

cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/Bayes/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/Bayes/output/output_without_cov.tsv"

simulated_data = cwd_path+"/Bayes/data/simulated_data.tsv"
test_data = cwd_path+"/Bayes/data/simulated_data_test.tsv"
outcome_file = cwd_path+"/Bayes/output/oucome.txt"


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
                f.write('%s,%d\n' % (ID[i][0], label[i]))

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


if __name__ == "__main__":

    x_img_train, x_all_train, ID_train = get_data(simulated_data)

    x_img_test, x_all_test, ID_test = get_data(test_data)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    gnb = GaussianNB()
    # With covariate
    label_with_cov = gnb.fit(x_all_train, true_label).predict(x_all_test)

    write_outputfile(output_file_with_cov, ID_test, label_with_cov,
                     true_label, outcome_file, "Hierachical with covariate")

    # Without covariate
    label_without_cov = gnb.fit(x_img_train, true_label).predict(x_img_test)

    write_outputfile(output_file_without_cov, ID_test, label_without_cov,
                     true_label, outcome_file, "Hierachical without covariate")
