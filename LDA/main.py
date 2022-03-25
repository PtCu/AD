from tkinter import N
from typing import Counter
import lda
import lda.datasets
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

cwd_path = os.getcwd()

output_file = cwd_path+"/LDA/output/output.tsv"

test_data = cwd_path+"/LDA/data/simulated_data.tsv"

outcome_file = cwd_path+"/LDA/output/oucome.txt"

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
            ID = data[:, np.nonzero(header == 'ID')[0]]
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
        out_label = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(numpy.int)

    measure = ARI(true_label, out_label)

    with open(outcome_file, 'a') as f:
        f.write(name+" :\n")
        f.write("ARI: " + str(measure)+'\n')


if __name__ == "__main__":

    feat_cov, feat_img, ID, group = read_data()

    feat_img = np.transpose(feat_img)

    x_img = np.transpose(feat_img[:, group == 1])  # patients

    x_img_flatten = x_img.flatten()
    count_data = Counter(x_img_flatten)
    X = np.zeros([PT_NUMS, ROI_NUMS], dtype=int)

    for i in range(PT_NUMS):
        for j in range(ROI_NUMS):
            X[i][j] = count_data[x_img[i][j]]

    model = lda.LDA(n_topics=2, n_iter=150, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    subtypes = {}

    n_top_words = ROI_NUMS  # 选取概率最大的20个ROI作为亚型的主题
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(x_img_flatten)[
            np.argsort(topic_dist)][:-(n_top_words+1):-1]
        with open("pattern.txt", 'a') as f:
            f.write('Type {}: {}'.format(i, topic_words))
            subtypes[i] = topic_words

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    out_label = np.zeros(PT_NUMS)

    for i in range(PT_NUMS):
        ROIS = x_img[i]
        out_label[i] = get_subtype(ROIS, subtypes, K)

    write_outputfile(output_file, ID, out_label,
                     true_label, outcome_file, "LDA")
