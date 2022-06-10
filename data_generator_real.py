
# 真实数据的读取和处理
# 输入为时间序列文件，输出为tsv或csv文件
# data2用于格式2，给HYDRA算法用的

import numpy as np

import os

import pandas as pd


K_min = 2
K_max = 9

cwd_path = os.getcwd()
data_dir = cwd_path+"/data/"
dirs = ['S01_mean_ts','S02_mean_ts','S03_mean_ts','S04_mean_ts','S05_mean_ts','S06_mean_ts','S07_mean_ts']
source_list = []
prefix = data_dir+"origin_data/"

for dir in dirs:
    for f_name in os.listdir(prefix+dir):
        if f_name.startswith('3'):
            source_list.append(prefix+dir+"/"+f_name)


NC_TYPE = 0
# 病号标签
PT_TYPE = 1
title1 = ["SET", "GROUP"]
title2 = ["session_id", "diagnosis"]
session_id = 0

# DEST_DIM=200 #降到200维
def add_label(X, title, dest_file):
    # x_t=decomposition.PCA(n_components=DEST_DIM).fit_transform(X)
    df = pd.DataFrame(X, columns=title)
    if "SET" in title:
        id_name = "ID"
    else:
        id_name = "participant_id"
    df.index.name = id_name
    df.to_csv(dest_file, sep='\t')
    return



def generate_from_one(X, data):
    synthetic_sample = X.T
    A = np.corrcoef(synthetic_sample)
    item = []
    # SET
    item.append(int(session_id))
    # GROUP

    item.append(int(PT_TYPE))
    for r in range(len(A)):
        for l in range(len(A[0])):
            item.append(A[r][l])

    data.append(item)

        

def generate_all():
    # 对于每个聚类数目
    X = pd.read_csv(source_list[0]).values
    for idx in range(X.shape[1]*X.shape[1]):
        title1.append("ROI")
        title2.append("ROI"+str(idx+1))
    data = []
    dest_file = data_dir+"real_data"
    for src in source_list:
        X=pd.read_csv(src, sep='\t|,|;').values
        generate_from_one(X, data)     

    add_label(data, title1, dest_file+"_real"+"_1.csv")
    add_label(data, title2, dest_file+"_real"+"_2.csv")
 




if __name__ == "__main__":
    generate_all()
