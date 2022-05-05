


from Kmedians.KMedianClusterer import KMedianClusterer
from Hierachical.HierachicalClusterer import HierachicalClusterer

from URF.URFClusterer import URFClusterer

import utilities.utils as utl
import os
import sys


sys.path.append(os.getcwd())
cwd_path = os.getcwd()

synthetic_data1 = cwd_path+"/data/clustering2_1.csv"
synthetic_data2 = cwd_path+"/data/synthetic_data2.csv"
simulated_data2 = cwd_path+"/data/feature.tsv"
simulated_data1 = cwd_path+"/data/simulated_data1.tsv"
real_data1 = cwd_path+"/data/real_data_real_1.csv"
real_data2 = cwd_path+"/data/real_data2.csv"

output_dir = cwd_path+"/output/"

K_min = 2
K_max = 5

if __name__ == "__main__":
    pt_nc_img, pt_nc_cov, set, ID, group = utl.get_data(
        real_data1)
    X = {}
    X["pt_nc_img"] = pt_nc_img
    X["pt_nc_cov"] = pt_nc_cov
    X["pt_ID"] = ID
    X["group"] = group

    # utl.plot_K(X, 2, 10, cwd_path+"/"+name+"/output/"+name,
    #            CHIMERAClusterer, title=label)

    # utl.get_ari(X,K_min,K_max,URFClusterer,KMedianClusterer,output_dir,"urf-k_medians")


    utl.get_matrix(X,K_min,K_max,URFClusterer,HierachicalClusterer,output_dir,"urf-Hierachical_cf_matrix")

    utl.get_matrix(X,K_min,K_max,KMedianClusterer,HierachicalClusterer,output_dir,"K_medians-Hierachical_cf_matrix")

    utl.get_matrix(X,K_min,K_max,URFClusterer,KMedianClusterer,output_dir,"urf-k_medians_matrix_cf_matrix")