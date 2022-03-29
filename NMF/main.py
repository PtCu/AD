
from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from sklearn.decomposition import NMF
import numpy as np
from scipy.spatial.distance import pdist
from typing import Counter
cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/NMF/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/NMF/output/output_without_cov.tsv"

simulated_data = cwd_path+"/NMF/data/simulated_data.tsv"

outcome_file = cwd_path+"/NMF/output/oucome.txt"

K = 2

PT_NUMS = 500
ROI_NUMS = 20


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

        return feat_cov, np.around(feat_img, decimals=3), ID, group


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


def jaccard_distance(x, y):
    """
    计算两个无序向量之间的欧拉距离，支持多维
    """
    X = np.vstack([x, y])
    d2 = pdist(X, 'jaccard')
    return d2[0]


def get_subtype(ROIS, subtypes, k):
    ans = 0
    min_dis = float('inf')
    for i in range(k):
        subtype = subtypes[i]
        dis = jaccard_distance(ROIS, subtype)
        if dis < min_dis:
            min_dis = dis
            ans = i
    return ans


def clustering1(X, k):
    model = NMF(n_components=k)
    A = model.fit_transform(X)  

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


def clustering2(X, k, x_img_flatten):
    # n_component 表示多少样本被保留
    model = NMF(n_components=k)
    W = model.fit_transform(X)  # model.fit_transform(X) is also available
    topic_word = model.components_  # model.components_ also works

    subtypes = {}
    n_top_words = ROI_NUMS  # 选取概率最大的20个ROI作为亚型的主题
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(x_img_flatten)[
            np.argsort(topic_dist)][:-(n_top_words+1):-1]
        with open("pattern.txt", 'a') as f:
            f.write('Type {}: {}'.format(i, topic_words))
            subtypes[i] = topic_words

    out_label = np.zeros(PT_NUMS)
    for i in range(PT_NUMS):
        ROIS = x_img[i]
        out_label[i] = get_subtype(ROIS, subtypes, K)
    return out_label


if __name__ == "__main__":

    x_img, x_all, ID = get_data(simulated_data)

    x_img_flatten = x_img.flatten()
    count_data = Counter(x_img_flatten)
    X = np.zeros([PT_NUMS, ROI_NUMS], dtype=int)

    for i in range(PT_NUMS):
        for j in range(ROI_NUMS):
            X[i][j] = count_data[x_img[i][j]]

    label = clustering1(X, K)

    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    # Without covariate
    write_outputfile(output_file_without_cov, ID, label,
                     true_label, outcome_file, "Hierachical without covariate")
