# 仿真数据1的生成
# data2用于格式2，给HYDRA算法用的

from operator import imod
import numpy as np
import csv

from pip import main
from pandas import DataFrame, Series

import pandas as pd
import os

TOTAL_NUM = 500

# 正常人标签
NC_TYPE = 0
# 病号标签
PT_TYPE = 1

DATA_SET = 0

cwd_path = os.getcwd()

data_dir = cwd_path+"/data/"

dest_file1 = os.path.join(data_dir, "simulated_data1.tsv")
dest_file2 = os.path.join(data_dir, "simulated_data2.tsv")



# COVAR1为年龄，COVAR2为性别
sample_id = int(0)

# 格式1
data1 = []

# 格式2. 这里为了方便直接冗余了一份。
data2 = []
session_id=0

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def gen_a_nc_sample(isNomalized=True):
    item1 = []
    item2=[]
    global sample_id
    sample_id += 1
    volume_size = np.random.normal(1, 0.1, 20)
    age = int(np.random.uniform(55.0, 85.0))
    atrophy = np.random.normal(0.01*(age-55), 0.005*(age-55), 20)
    sex = np.random.randint(0, 2)
    # 随着年龄的自然萎缩作用量
    volume_size -= atrophy
    # 归一化
    if isNomalized:
        volume_size = normalization(volume_size)

    item1.append(sample_id)
    item1.append(NC_TYPE)
    item1.append(DATA_SET)
    item1.append(age)
    item1.append(sex)
    item1.extend(volume_size)
    data1.append(item1)

    item2.append(sample_id)
    item2.append(session_id)
    item2.append(NC_TYPE)
    item2.extend(volume_size)
    data2.append(item2)




def gen_a_pt_sample(type, isNomalized=True):
    item1 = []
    item2=[]
    global sample_id
    sample_id += 1
    volume_size = np.random.normal(1, 0.1, 20)
    age = int(np.random.uniform(55.0, 85.0))
    atrophy = np.random.normal(0.01*(age-55), 0.005*(age-55), 20)
    sex = np.random.randint(0, 2)
    # 随着年龄的自然萎缩作用量
    volume_size -= atrophy
    # 归一化
    if isNomalized:
        volume_size = normalization(volume_size)


    # 添加人为标记的萎缩量
    # 前250个为1型
    if(type == 0):
        # 0到9号ROI区域标记为萎缩
        for i in range(0, 9):
            volume_size[i] *= 0.85
    # 后250个为2型
    else:
        # 0,1,5,6,10,11,15,16号区域标记为萎缩
        for i in range(0, 4):
            volume_size[i*5] *= 0.85
            volume_size[i*5+1] *= 0.85

    item1.append(sample_id)
    item1.append(PT_TYPE)
    item1.append(DATA_SET)
    item1.append(age)
    item1.append(sex)
    item1.extend(volume_size)
    data1.append(item1)

    item2.append(sample_id)
    item2.append(session_id)
    item2.append(PT_TYPE)
    item2.extend(volume_size)
    data2.append(item2)


def gen_samples():
    for i in range(0, TOTAL_NUM):
        gen_a_nc_sample()

    for i in range(0, TOTAL_NUM//2):
        gen_a_pt_sample(0)

    for i in range(0, TOTAL_NUM//2):
        gen_a_pt_sample(1)

title1 = ["ID", "GROUP", "SET", "COVAR", "COVAR"]
title2 = ["participant_id", "session_id", "diagnosis"]

if __name__ == "__main__":

    with open(dest_file1, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in range(1, 21):
            title1.append("ROI")
            
        tsv_w.writerow(title1)
    
    with open(dest_file2, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in range(0, 20):
            title2.append("ROI"+str(i+1))
            
        tsv_w.writerow(title2)

    gen_samples()
    df1 = pd.DataFrame(data1, columns=title1)
    final_df = df1.set_index("ID")

    final_df.to_csv(dest_file1, sep='\t')

    df2 = pd.DataFrame(data2, columns=title2)
    final_df = df2.set_index("participant_id")

    final_df.to_csv(dest_file2, sep='\t')


