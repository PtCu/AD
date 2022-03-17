from pdb import main

import chimera_clustering
import pickle
import os
import sys
import csv
import numpy
from sklearn.metrics import adjusted_rand_score as ARI
import pandas as pd
cwd_path = os.getcwd()
feature_tsv = cwd_path+"/CHIMERA/data/test_feature.tsv"
output_dir = cwd_path+"/CHIMERA/output"
k = 2
covariate_tsv = cwd_path+"/CHIMERA/data/test_covariate.tsv"

model_file = output_dir+'/clustering_model.pkl'

test_data = cwd_path+"/CHIMERA/data/simulated_data.tsv"


if __name__ == "__main__":
    # chimera_clustering.clustering(test_data, output_dir,k)
    # f = open(model_file,'rb')
    # data = pickle.load(f)
    # print(data)

    with open(output_dir+'/output.tsv') as f:
        out_label = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(numpy.int)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    measure = ARI(true_label, out_label)

    sys.stdout.write("Test Complete, output labels in test/ folder.\n")
    sys.stdout.write(
        "Clustering test samples yields an adjusted rand index of %.3f with ground truth labels.\n" % measure)
    if measure >= 0.9:
        sys.stdout.write("Test is successful.\n")

    with open(output_dir+'/outcome.txt', 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
        f.write("ARI: "+str(measure))
