# 仿真数据2的生成
# data2用于格式2，给HYDRA算法用的

from dataclasses import replace
from random import randint, random
import numpy as np
from pip import main
from pyparsing import col
from sklearn.utils import check_random_state, check_array
import utilities.utils as utl
import os
import sys
import csv
import pandas as pd


K_min = 2
K_max = 9

cwd_path = os.getcwd()
data_dir = cwd_path+"/data/"

source_list = []
prefix = data_dir+"origin_data/S01_mean_ts/1_NC00"
for i in range(1, 10):
    source_list.append(prefix+str(i)+"_ts.csv")

sample_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
NC_TYPE = 0
# 病号标签
PT_TYPE = 1
title1 = ["SET", "GROUP"]
title2 = ["session_id", "diagnosis"]
session_id = 0


def generate_all():

    X = pd.read_csv(source_list[0]).values
    for idx in range(X.shape[1]*(X.shape[1]-1)//2):
        title1.append("ROI")
        title2.append("ROI"+str(idx+1))

    # 时间序列长度为随机数
    for k in range(K_min, K_max):
        t=randint(120,150)
        # 取k个样本数据
        data = []
        dest_file = data_dir+"clustering"+str(k)
        random_state = check_random_state(k)
        selected_sample = random_state.choice(
            sample_list, size=k, replace=False)
        for j in range(len(selected_sample)):
            selected_file = source_list[selected_sample[j]]
            X = pd.read_csv(selected_file).values

            generate_from_one(X, data, j, t)

        add_label(data, title1, dest_file+"_random_1.csv")
        add_label(data, title2, dest_file+"_random_2.csv")

    # 时间序列长度为120，130，140，150
    for t in range(120, 151, 10):
        for k in range(K_min, K_max):
            # 取k个样本数据
            data = []
            dest_file = data_dir+"clustering"+str(k)
            random_state = check_random_state(k)
            selected_sample = random_state.choice(
                sample_list, size=k, replace=False)
            for j in range(len(selected_sample)):
                selected_file = source_list[selected_sample[j]]
                X = pd.read_csv(selected_file).values

                generate_from_one(X, data, j, t)

            add_label(data, title1, dest_file+"_"+str(t)+"_1.csv")
            add_label(data, title2, dest_file+"_"+str(t)+"_2.csv")

   

def add_label(X, title, dest_file):

    df = pd.DataFrame(X, columns=title)
    if "SET" in title:
        id_name = "ID"
    else:
        id_name = "participant_id"
    df.index.name = id_name
    df.to_csv(dest_file, sep='\t')
    return


def generate_from_one(X, data, idx, t):
    m = X.shape[0]-t
    # 一个原始样本生成m个仿真数据
    for i in range(m):
        random_state = check_random_state(i)
        start_idx = random_state.choice(m, size=1, replace=False)
        synthetic_sample = X[start_idx[0]:start_idx[0]+t].T
        # 相关系数
        A = np.corrcoef(synthetic_sample)
        item = []
        # SET
        item.append(int(session_id))
        # GROUP
        if idx == 0:
            item.append(int(NC_TYPE))
        else:
            item.append(int(PT_TYPE))
 
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                item.append(A[i][j])

        data.append(item)


if __name__ == "__main__":
    generate_all()
