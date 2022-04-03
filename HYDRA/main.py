from mlni.hydra_clustering import clustering
import os
import numpy as np


cwd_path = os.getcwd()

covariate_tsv = cwd_path+"/HYDRA/data/covariate.tsv"

feature_tsv = cwd_path+"/HYDRA/data/feature.tsv"

output_dir = cwd_path + "/HYDRA/output"

k_min = 2
k_max = 2
cv_repetition = 200
true_label = np.append(np.zeros(250), np.ones(250))

clustering(feature_tsv, output_dir, k_min, k_max,
           cv_repetition, true_label, covariate_tsv=covariate_tsv)
