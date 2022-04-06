from pdb import main

import chimera.chimera_clustering
import pickle
import os
import csv,sys
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

cwd_path = os.getcwd()
feature_tsv = cwd_path+"/CHIMERA/data/test_feature.tsv"
output_dir = cwd_path+"/CHIMERA/output"
k = 2
covariate_tsv = cwd_path+"/CHIMERA/data/test_covariate.tsv"

model_file = output_dir+'/clustering_model.pkl'

simulated_data = cwd_path+"/CHIMERA/data/simulated_data.tsv"
output_dir = cwd_path + "/CHIMERA/output/"


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

def eval_K(data_file, k_min, k_max, filename="outcome.png"):
    feat_cov, feat_img, ID, group = read_data(data_file)
    x = np.arange(k_min, k_max+1, dtype=int)
    y = []
    for k in range(k_min, k_max+1):
        # ck=cluster.KMeans(k)
        # label=ck.fit(X).labels_
        label, X= chimera.chimera_clustering.clustering(
            feat_cov,feat_img,ID,group, k, transformation_type='affine')
        silhouette_avg = silhouette_score(X, label)
        y.append(silhouette_avg)
    plt.title("CHIMERA")
    plt.xlabel("K range")
    plt.ylabel("Silhoutte score")
    plt.plot(x, y)
    # plt.show()
    plt.savefig(output_dir+filename)
    plt.clf()

def cluster(type='affine'):
    config = chimera.chimera_clustering.clustering(
        simulated_data, output_dir, k, transformation_type='affine')
    f = open(model_file, 'rb')
    data = pickle.load(f)
    print(data)

    with open(output_dir+'/output.tsv') as f:
        out_label = np.asarray(list(csv.reader(f)))
        idx = np.nonzero(out_label[0] == "Cluster")[0]
        out_label = out_label[1:, idx].flatten().astype(np.int)
        true_label = np.append(np.ones(250), np.ones(250)*2)
        measure = ARI(true_label, out_label)
        with open(output_dir+'/outcome.txt', 'a') as f:
            f.write("Config:\n")
            f.write(str(config))
            f.write("\n"+type+" ARI: " + str(measure))

if __name__ == "__main__":
    # cluster('affine')
    # cluster('duo')
    # cluster('trans')
    eval_K(simulated_data,2,8)

