from pdb import main

import chimera_clustering

import os,sys

cwd_path=os.getcwd()
feature_tsv=cwd_path+"/CHIMERA/data/test_feature.tsv"
output_dir = cwd_path+"/CHIMERA/output"
k =2
covariate_tsv=cwd_path+"/CHIMERA/data/test_covariate.tsv"


if __name__=="__main__":
    chimera_clustering.clustering(feature_tsv, output_dir, k, covariate_tsv)