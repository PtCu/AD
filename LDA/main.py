from typing import Counter
import lda
import lda.datasets
from pdb import main
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.naive_bayes import ComplementNB

cwd_path = os.getcwd()

output_file = cwd_path+"/LDA/output/output.tsv"

simulated_data = cwd_path+"/LDA/data/simulated_data.tsv"

outcome_file = cwd_path+"/LDA/output/oucome.txt"

output_dir = cwd_path + "/LDA/output/"

K = 2
TOTAL_NUMS=1000
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

    return feat_cov, np.around(feat_img, decimals=3), ID, group


def jaccard_distance(x, y):
    """
    计算两个无序向量之间的欧拉距离，支持多维
    """
    X = np.vstack([x, y])
    d2 = pdist(X, 'jaccard')
    return d2[0]

# 用某种方法来度量两个向量的相似度，匹配到最为相似的那个subtype上


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


def clustering1(X, k):
    model = lda.LDA(n_topics=k, n_iter=150, random_state=1)
    A = model.fit_transform(X)  # model.fit_transform(X) is also available

    label = np.zeros(TOTAL_NUMS, dtype=int)

    for i in range(len(A)):
        subtype = 0
        max_prob = 0
        for j in range(k):
            if(A[i][j] > max_prob):
                max_prob = A[i][j]
                subtype = j
        label[i] = subtype
    return label


def clustering2(X, k, x_img_flatten,):
    model = lda.LDA(n_topics=k, n_iter=150, random_state=1)
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

    out_label = np.zeros(PT_NUMS)
    for i in range(PT_NUMS):
        ROIS = x_img[i]
        out_label[i] = get_subtype(ROIS, subtypes, K)
    return out_label


def eval_K(X, k_min, k_max, filename):
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        label = clustering1(X, k)
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("LDA")
    plt.ylabel("Silhoutte score")
    plt.xlabel("K range")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+filename)
    plt.clf()


def test(source, k_min, k_max, name):
    source_flatten = source.flatten()
    count_data = Counter(source_flatten)
    X = np.zeros([len(source), len(count_data)], dtype=int)
    for i in range(len(source)):
        for j in range(len(source[0])):
            X[i][j] = count_data[source[i][j]]

    eval_K(X, k_min, k_max, name)


if __name__ == "__main__":

    x_img, x_all, feat_img, feat_all, ID = get_data(simulated_data)

    test(feat_img, 2, 30, "PT_NC.png")
    # test(x_img, 2, 10, "only_PT.png")
    # out_label = clustering1(X, K)

    # true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    # write_outputfile(output_file, ID, out_label,
    #                  true_label, outcome_file, "LDA")
