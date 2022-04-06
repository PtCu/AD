from pdb import main

import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from sklearn import cluster
import random
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

cwd_path = os.getcwd()

output_file_with_cov = cwd_path+"/K-medians/output/output_with_cov.tsv"

output_file_without_cov = cwd_path+"/K-medians/output/output_without_cov.tsv"

simulated_data = cwd_path+"/K-medians/data/simulated_data.tsv"

outcome_file = cwd_path+"/K-medians/output/oucome.txt"

output_dir = cwd_path + "/K-medians/output/"

K = 2


class KMedianClusterer:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.x_data =[]


    def fit(self,X):
        self.x_data = X["pt_nc_img"]
        self.points = self.__pick_start_point(self.x_data, self.cluster_num)
        self.labels_=np.zeros(len(self.x_data))
        self.__cluster()

    def __cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([self.x_data[i]]) #选前cluster_num个点初始化
        for idx, item in enumerate(self.x_data):
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance <= distance_min:
                    distance_min = distance
                    index = i
            self.labels_[idx] = index
            result[index] = result[index] + [item.tolist()]

        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            return 

        self.points = np.array(new_center)
        return self.__cluster()

    def __center(self, list):
        '''计算一组坐标的中心点
        '''
        # 计算每一列的中位数
        return np.median(list, axis=0)

    def __distance(self, p1, p2):
        '''计算两点间距
        '''
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i]-p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):

        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")

        # 随机点的下标
        indexes = random.sample(
            np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)


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
    cluster.KMeans()
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
    X = np.array(X)
    clusterer = KMedianClusterer(X, k)
    label = clusterer.cluster()
    return label


def eval_K(X, k_min, k_max, filename="outcome.png"):
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        # ck=cluster.KMeans(k)
        # label=ck.fit(X).labels_
        label = clustering(X, k)
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("K-medians")
    plt.xlabel("K range")
    plt.ylabel("Silhoutte score")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+filename)
    plt.clf()


if __name__ == "__main__":

    x_img, x_all, feat_img, feat_all, ID = get_data(simulated_data)

    true_label = numpy.append(numpy.zeros(250), numpy.ones(250))

    eval_K(feat_img, 2, 100)


