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
    config=chimera_clustering.clustering(test_data, output_dir, k)
    f = open(model_file, 'rb')
    data = pickle.load(f)
    print(data)

    with open(output_dir+'/output.tsv') as f:
        out_label = numpy.asarray(list(csv.reader(f)))

    idx = numpy.nonzero(out_label[0] == "Cluster")[0]
    out_label = out_label[1:, idx].flatten().astype(numpy.int)

    true_label = numpy.append(numpy.ones(250), numpy.ones(250)*2)

    measure = ARI(true_label, out_label)

    with open(output_dir+'/outcome.txt','a') as f:
        f.write("Config:\n")
        f.write(str(config))
        f.write("\nARI1: " + str(measure))

