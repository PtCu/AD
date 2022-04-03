from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster


cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/hierachical/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/hierachical/output/output_without_cov.tsv"

simulated_data = cwd_path+"/hierachical/data/simulated_data.tsv"

output_dir = cwd_path + "/hierachical/output/"

outcome_file = cwd_path+"/hierachical/output/oucome.txt"

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

    x_img = feat_img[group == 1, :]  # patients

    x_all = feat_all[group == 1, :]  # patients

    return x_img, x_all, feat_img, feat_all, ID


def clustering(X, k):
    sk = cluster.AgglomerativeClustering(k)
    sk.fit(X)
    label = sk.labels_
    return label


def eval_K(X, k_min, k_max):
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        label = clustering(X, k)
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("silhouette score")
    # fontproperties 设置中文显示，fontsize 设置字体大小
    plt.xlabel("Silhoutte score")
    plt.ylabel("K range")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+'outcome.png')


def eval_K(X, k_min, k_max, filename="outcome.png"):
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        label = clustering(X, k)
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("silhouette score")
    plt.xlabel("Silhoutte score")
    plt.ylabel("K range")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+filename)
    plt.clf()


if __name__ == "__main__":

    x_img, x_all, ID, feat_img, feat_all = get_data(simulated_data)

    eval_K(feat_img, 2, 8)
    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    # # Without covariate
    # label_without_cov = clustering(x_img)
    # write_outputfile(output_file_without_cov, ID, label_without_cov,
    #                  true_label, outcome_file, "Hierachical without covariate")
