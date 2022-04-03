import numpy as np
import csv

from pip import main
from pandas import DataFrame, Series

import pandas as pd

TOTAL_NUM = 500

#正常人标签
NC_TYPE = -1
#病号标签
PT_TYPE = 1

DATA_SET = 0

file_name = "simulated_data.tsv"
feature_file="feature.tsv"
covariate_file="covariate.tsv"
#COVAR1为年龄，COVAR2为性别
sample_id = int(0)

features = []
covariates=[]
session_id="ses-M00"

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def gen_a_nc_sample():
    feature = []
    covariate=[]
    global sample_id
    sample_id += 1
    volume_size = np.random.normal(1, 0.1, 20)
    age = int(np.random.uniform(55.0, 85.0))
    atrophy = np.random.normal(0.01*(age-55), 0.005*(age-55), 20)
    sex = np.random.randint(0, 2)
    #随着年龄的自然萎缩作用量
    volume_size -= atrophy
    #归一化
    volume_size = normalization(volume_size)
    #type为0时分开保存feature和covariate
    feature.append(sample_id)
    feature.append(session_id)
    feature.append(NC_TYPE)
    feature.extend(volume_size)
    features.append(feature)

    covariate.append(sample_id)
    covariate.append(session_id)
    covariate.append(NC_TYPE)
    covariate.append(age)
    covariate.append(sex)
    covariates.append(covariate)


def gen_a_pc_sample(type):
    feature = []
    covariate=[]
    global sample_id
    sample_id += 1
    volume_size = np.random.normal(1, 0.1, 20)
    age = int(np.random.uniform(55.0, 85.0))
    atrophy = np.random.normal(0.01*(age-55), 0.005*(age-55), 20)
    sex = np.random.randint(0, 2)
    #随着年龄的自然萎缩作用量
    volume_size -= atrophy
    #归一化
    volume_size = normalization(volume_size)

    #添加人为标记的萎缩量
    #前250个为1型
    if(type == 0):
        #0到9号ROI区域标记为萎缩
        for i in range(0, 9):
            volume_size[i] *= 0.85
    #后250个为2型
    else:
        #0,1,5,6,10,11,15,16号区域标记为萎缩
        for i in range(0, 4):
            volume_size[i*5] *= 0.85
            volume_size[i*5+1] *= 0.85

    feature.append(sample_id)
    feature.append(session_id)
    feature.append(PT_TYPE)
    feature.extend(volume_size)
    features.append(feature)

    covariate.append(sample_id)
    covariate.append(session_id)
    covariate.append(PT_TYPE)
    covariate.append(age)
    covariate.append(sex)
    covariates.append(covariate)

    # with open(file_name, 'a',newline='') as f:
    #     tsv_w = csv.writer(f, delimiter='\t')
    #     l=np.array([int(sample_id),PT_TYPE, age, sex])
    #     a_row=np.append(l,volume_size)
    #     tsv_w.writerow(a_row)


def gen_samples():
    for i in range(0, TOTAL_NUM):
        gen_a_nc_sample()

    for i in range(0, TOTAL_NUM//2):
        gen_a_pc_sample(0)

    for i in range(0, TOTAL_NUM//2):
        gen_a_pc_sample(1)


if __name__ == "__main__":

    with open(feature_file, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title_feature = ["participant_id", "session_id", "diagnosis"]
        for i in range(1, 21):
            title_feature.append("ROI"+str(i))

        tsv_w.writerow(title_feature)

    with open(covariate_file, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        title_covariate = ["participant_id", "session_id", "diagnosis","age","sex"]

        tsv_w.writerow(title_covariate)

    gen_samples()
    df = pd.DataFrame(features, columns=title_feature)
    feature_df = df.set_index("participant_id")

    # 保存 dataframe
    feature_df.to_csv(feature_file, sep='\t')

    df = pd.DataFrame(covariates, columns=title_covariate)
    covariate_df = df.set_index("participant_id")

    # 保存 dataframe
    covariate_df.to_csv(covariate_file, sep='\t')
